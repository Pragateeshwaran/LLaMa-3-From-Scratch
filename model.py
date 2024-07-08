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

# class LlamaForCausalLM(nn.Module):
#     def __init__(self)