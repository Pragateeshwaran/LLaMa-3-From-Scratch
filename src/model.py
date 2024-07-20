import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Config:
    def __init__(self):
        self.embedding_size = 4096
        self.vocab_size     = 128256
        self.block_count    = 32
        self.proj           = 14336
        self.eps            = 1e-5
        self.max_seq_len    = 2048
        self.num_attention_heads = 32
        self.head_dim       = self.embedding_size // self.num_attention_heads
        assert self.embedding_size % self.num_attention_heads == 0, "embedding_size must be divisible by num_attention_heads"

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.embedding_size, config.proj, bias=False)
        self.up_proj   = nn.Linear(config.embedding_size, config.proj, bias=False)
        self.down_proj = nn.Linear(config.proj, config.embedding_size, bias=False)
        self.act_fn    = nn.SiLU()

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class LlamaRMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            self.cos_cached = None
            self.sin_cached = None
        if self.cos_cached is None or self.sin_cached is None:
            t = torch.arange(self.max_seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[:seq_len]
            self.sin_cached = emb.sin()[:seq_len]
        return (
            self.cos_cached.unsqueeze(0),
            self.sin_cached.unsqueeze(0),
        )

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaSdpaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        
        self.q_proj = nn.Linear(config.embedding_size, config.embedding_size, bias=False)
        self.k_proj = nn.Linear(config.embedding_size, config.embedding_size, bias=False)
        self.v_proj = nn.Linear(config.embedding_size, config.embedding_size, bias=False)
        self.o_proj = nn.Linear(config.embedding_size, config.embedding_size, bias=False)
        
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)

    def forward(self, hidden_states ):
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(q, seq_len=q_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True) 
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        
        return self.o_proj(attn_output)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaSdpaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.embedding_size, config.eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.embedding_size, config.eps)

    def forward(self, hidden_states ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embedding_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.block_count)])
        self.norm = LlamaRMSNorm(config.embedding_size, config.eps)

    def forward(self, input_ids ):
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)

        return hidden_states

class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size, bias=False)

    def forward(self, input_ids, target=None):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        if target is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return loss

    def generate(self, input_ids, max_length, temperature=1.0, top_k=50, top_p=0.95):
        for _ in range(max_length - input_ids.size(1)):
            outputs = self(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1).expand_as(logits)
        logits = torch.where(logits < min_values, torch.full_like(logits, filter_value), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits

def Model_loader():
    config = Config()
    model = LlamaForCausalLM(config)
    return model, sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example usage:
# model, num_params = Model_loader()
# print(f"Number of parameters: {num_params}")

# Generate some random input
# input_ids = torch.randint(0, config.vocab_size, (1, 10))

# Forward pass
# logits = model(input_ids)
# print(f"Output logits shape: {logits.shape}")

# Generate text
# generated = model.generate(input_ids, max_length=20)
# print(f"Generated sequence shape: {generated.shape}")