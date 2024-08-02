import transformers
import torch
from huggingface_hub import login

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
cache_dir = r'D:\\hugging-models\\llama3-meta-pragateesh'
login(token='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')

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

def encoder(input : str) -> list: 
    return tokenizer.encode(input, add_special_tokens=False) 

def decoder(input : list) -> str:
    return tokenizer.decode(input)

def eot():
    return tokenizer.eos_token_id
