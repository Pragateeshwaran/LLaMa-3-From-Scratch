import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import time 
import model as Model
import numpy as np 
import os 
import math
from Tokenizers import encoder, decoder

model = Model.load_model()
torch.set_float32_matmul_precision('high')
device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else "cpu"
# Model = Model.to(device)

#------------------------configurations--------------------------- 
max_epochs = 100000     
max_lr = 3e-4           # as per paper
min_lr = max_lr * 0.1   # as per paper
warmup_steps = 2000     # as per paper
weight_decay = 0.1      # as per paper
Opti_Beta1 = 0.9        # as per paper
Opti_Beta2 = 0.5        # as per paper
Opti_epi = 1e-5         # as per paper
B = 16                  # as per GPU capacity
T = 512                 # as per GPU capacity
total_batch_size = 524288
assert total_batch_size % (B*T) == 0, "make sure total batch size is divisible by B*T"
grad_accum_steps = total_batch_size // (B*T)
print(f"Total Batch size is {total_batch_size}")
print(f"Gram accumulation is {grad_accum_steps}")

#-------------------------Data Loader-----------------------------
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}
        data_root =r'F:\works\A-important\A-neurals\LLaMa-3-From-Scratch\src\Dataset'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x.to(device), y.to(device)

train_loader = DataLoaderLite(B, T, 'train')
val_loader = DataLoaderLite(B, T, 'val')
# ------------------------------CheckPoints--------------------------------------------
log_dir = r"F:\works\A-important\A-neurals\LLaMa-3-From-Scratch\logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "training_log.txt")

# ------------------------------Optimizers----------------------------------------------
def get_lr(it): 
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps 
    if it > max_epochs:
        return min_lr 
    decay_ratio = (it - warmup_steps) / (max_epochs - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay= weight_decay, learning_rate= max_lr, b1= Opti_Beta1, b2= Opti_Beta2, eps= Opti_epi)

print(optimizer)
val_lossi, trainl_lossi = [], []

for iter in range(max_epochs):
    print(".", end= " ")
    last_step = (iter == (max_epochs - 1))
    time_start = time.time()
    if iter % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_accum = 0
            val_steps = 20
            for _ in range(val_steps):
                x, y = val_loader.next_batch()
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_steps
                val_accum += loss.detach()
        print(f"validation loss: {val_accum.item():.4f}")
        val_lossi.append(val_accum.item())
        with open(log_file, "a") as f:
            f.write(f"{iter} val {val_accum.item():.4f}\n")
        if iter > 0 and (iter % 5000 == 0 or last_step):
            checkpoint_path = os.path.join(log_dir, f"model_{iter:05d}.pt")
            checkpoint = {
                'model': model.state_dict(),
                'config': model.config,
                'step': iter,
                'val_loss': val_accum.item()
            }
            torch.save(checkpoint, checkpoint_path)

    if (iter > 0 and iter % 250 == 0) or last_step or (iter == 0):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = encoder("What really peace means")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, _ = model(xgen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = decoder(tokens)
            print(f"\nsample {i}: {decoded}\n")

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    time_end = time.time()
    dt = time_end - time_start
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    trainl_lossi(loss_accum.item())
    print(f"step {iter:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    with open(log_file, "a") as f:
        f.write(f"{iter} train {loss_accum.item():.6f}\n")
        
