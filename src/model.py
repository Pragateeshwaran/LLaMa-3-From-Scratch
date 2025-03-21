import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

# Set device to CPU by default; uncomment to use GPU if available
device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model configuration class
class ModelConfig:
    def __init__(self):
        self.vocab_size = 128256  # Vocabulary size
        self.dim = 4096           # Model dimension
        self.n_layers = 32        # Number of transformer layers
        self.n_heads = 32         # Number of attention heads
        self.max_seq_len = 2048   # Maximum sequence length
        self.norm_eps = 1e-6      # Epsilon for normalization stability
        self.hidden_dim = 14336   # Hidden dimension for MLP

# RMS Normalization layer
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# Precompute frequencies for rotary embeddings
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=device)  # Ensure device consistency
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

# Apply rotary embeddings to query and key tensors
def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)
    
    freqs_cos = freqs_cos.view(1, freqs_cos.shape[0], 1, freqs_cos.shape[1])
    freqs_sin = freqs_sin.view(1, freqs_sin.shape[0], 1, freqs_sin.shape[1])
    
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

# Repeat key and value tensors for multi-query attention (if applicable)
def repeat_kv(x, n_rep):
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

# Attention mechanism
class LlamaAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_kv_heads = config.n_heads
        self.n_local_heads = config.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 1 in this case
        self.head_dim = config.dim // config.n_heads
        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, config.max_seq_len, config.max_seq_len), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(self, x, freqs_cos, freqs_sin):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None, dropout_p=0.0, is_causal=True
            )
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(output)

# MLP (Feed-Forward Network) layer
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# Decoder layer combining attention and MLP
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.self_attn(self.input_layernorm(x), freqs_cos, freqs_sin)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

# Main Llama model
class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.output.weight = self.embed_tokens.weight  # Weight tying
        freqs_cos, freqs_sin = precompute_freqs_cis(config.dim // config.n_heads, config.max_seq_len * 2)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, tokens, targets=None):
        batch_size, seqlen = tokens.shape
        h = self.embed_tokens(tokens)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)
        output = self.output(h)
        if targets is not None:
            logits = output[:, :-1, :].contiguous()
            targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return output, loss
        return output, None

    def configure_optimizers(self, weight_decay, learning_rate, b1, b2, eps):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        print(f"using fused AdamW: {use_fused}")
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(b1, b2), eps=eps, fused=use_fused)

# Utility function to compute total parameters
def compute_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Function to load and initialize the model
def load_model():
    config = ModelConfig()
    model = LlamaModel(config)
    model.to(device)  # Move model to the specified device
    total_params = compute_total_parameters(model)
    print(f"Total parameters: {total_params:,}")
    return model

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 2
    seq_len = 10
    
    # Load model
    model = load_model()
    
    # Create sample input
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    
    # Forward pass without targets
    output, loss = model(input_ids)
    print("Output shape:", output.shape)  # Should be [batch_size, seq_len, vocab_size]
    print("Loss:", loss)  # Should be None
    
    # Create sample targets for loss computation
    targets = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    output, loss = model(input_ids, targets)
    print("Output shape with targets:", output.shape)
    print("Loss with targets:", loss.item())
    
    # Configure optimizer
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=3e-4,
        b1=0.9,
        b2=0.95,
        eps=1e-8
    )
    print("Optimizer configured successfully")