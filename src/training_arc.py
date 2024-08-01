import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import time 
import model as Model
import numpy as np 
import os 
import math

model = Model.load_model()
torch.set_float32_matmul_precision('high')
device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else "cpu"
# Model = Model.to(device)

#------------------------configurations--------------------------- 
max_epochs = 100000     # as per andrej code
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
        data_root =r''
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

for iter in range(max_epochs):
    time_start = time.time()
    
    time_end   = time.time()
    print(f"Iter {iter} | Time {time_end - time_start} | ")