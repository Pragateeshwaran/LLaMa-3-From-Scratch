import os
import multiprocessing as mp
import numpy as np
import transformers
import torch
from huggingface_hub import login
from datasets import load_dataset
from tqdm import tqdm
from functools import partial

def tokenize(doc, eot, encoder):
    tokens = [eot]
    tokens.extend(encoder(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**32).all(), "token dictionary too large for uint32"
    tokens_np_uint32 = tokens_np.astype(np.uint32)
    return tokens_np_uint32

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def encoder(tokenizer, input: str) -> list:
    return tokenizer.encode(input, add_special_tokens=False)

def process_dataset(local_dir, remote_name, shard_size, hf_token, model_id, cache_dir):
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    try:
        # Load the dataset
        fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", cache_dir=DATA_CACHE_DIR)

        # Set up the tokenizer
        login(token=hf_token, add_to_git_credential=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

        eot = tokenizer.eos_token_id
        tokenize_partial = partial(tokenize, eot=eot, encoder=partial(encoder, tokenizer))

        nprocs = max(1, os.cpu_count()//2)
        with mp.Pool(nprocs) as pool:
            shard_index = 0
            all_tokens_np = np.empty((shard_size,), dtype=np.uint32)
            token_count = 0
            total_tokens = 0

            progress_bar = tqdm(total=len(fw), unit="documents", desc="Processing documents")

            for tokens in pool.imap(tokenize_partial, fw, chunksize=16):
                if token_count + len(tokens) < shard_size:
                    all_tokens_np[token_count:token_count+len(tokens)] = tokens
                    token_count += len(tokens)
                else:
                    split = "val" if shard_index == 0 else "train"
                    filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                    remainder = shard_size - token_count
                    all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                    write_datafile(filename, all_tokens_np)
                    shard_index += 1
                    all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                    token_count = len(tokens)-remainder

                total_tokens += len(tokens)
                progress_bar.update(1)

            if token_count != 0:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                write_datafile(filename, all_tokens_np[:token_count])

            progress_bar.close()

        print(f"Dataset processing completed. Total tokens processed: {total_tokens}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    mp.freeze_support()
    local_dir = r"F:\works\A-important\A-neurals\LLaMa-3-From-Scratch\src\Dataset"
    remote_name = "sample-10BT"
    shard_size = int(1e8)   
    hf_token = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    cache_dir = r'D:\hugging-models\llama3-meta-pragateesh'

    process_dataset(local_dir, remote_name, shard_size, hf_token, model_id, cache_dir)
