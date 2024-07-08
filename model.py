import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken 
from dataclasses import dataclass
import time 
import math 
import inspect 
import numpy as np 
import os  

# --------------------------------------------------------
class config():
    def __init__(self):
        self.embedding_size = 
class LlamaForCausalLM(nn.Module):
    def __init__(self):
        self.LlamaModel = LlamaModel(config)
        self.lm_head    = nn.Linear(config.embedding_size, config.vocab_size)