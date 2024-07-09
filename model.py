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
class Config:
    def __init__(self):
        self.embedding_size = 4096
        self.vocab_size     = 128256
        self.block_count    = 32
        self.proj           = 14336
        self.eps            = 1e-5

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(in_features=config.embedding_size, out_features=config.proj, bias=False)
        self.up_proj   = nn.Linear(in_features=config.embedding_size, out_features=config.proj, bias=False)
        self.down_proj = nn.Linear(in_features=config.proj, out_features=config.embedding_size, bias=False)
        self.act_fn    = nn.SiLU()

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
    def __init__(self):
        super().__init__()
        ...

class LlamaSdpaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(in_features=config.embedding_size, out_features=config.embedding_size, bias=False)
        self.k_proj = nn.Linear(in_features=config.embedding_size, out_features=config.embedding_size // 4, bias=False)
        self.v_proj = nn.Linear(in_features=config.embedding_size, out_features=config.embedding_size // 4, bias=False)
        self.o_proj = nn.Linear(in_features=config.embedding_size, out_features=config.embedding_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding()

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaSdpaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.embedding_size, config.eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.embedding_size, config.eps)

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embedding_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.block_count)])
        self.norm = LlamaRMSNorm(config.embedding_size, config.eps)

class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size, bias=False)

config = Config()
model = LlamaForCausalLM(config)