import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import time 
import model as Model
import numpy as np 
import os 

model, parameter_count = Model.load_model()
torch.set_float32_matmul_precision('high')
device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else "cpu"
# Model = Model.to(device)

#------------------------configurations--------------------------- 
max_epochs = 100000
max_lr = 3e-4

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
        data_root = r"F:\works\A-important\A-neurals\Vortex-Language-Models\GPT-2 From scratch\edu_fineweb10B"  
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


for iter in range(max_epochs):
    time_start = time.time()
    
    time_end   = time.time()
    print(f"Iter {iter} | Time {time_end - time_start} | ")