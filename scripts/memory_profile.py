import torch
import gc
from transformers import AutoModelForCausalLM, AutoConfig
from cggr import CGGRModel, create_truncated_router
import time

def get_peak_memory():
    return torch.cuda.max_memory_allocated() / 1024**2

def benchmark_memory(batch_size, mode='standard'):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM-135M")
    model = AutoModelForCausalLM.from_config(config).cuda()
    model.train()
    
    if mode == 'cggr':
        router = create_truncated_router(model, num_layers=4)
        model = CGGRModel(model, router=router, min_tokens_ratio=0.25, warmup_steps=0)

    # Inputs
    seq_len = 1024
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
    labels = torch.randint(0, 1000, (batch_size, seq_len)).cuda()

    try:
        # Forward + Backward
        if mode == 'cggr':
            loss = model(input_ids, labels)
        else:
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            
        loss.backward()
        
        peak_mem = get_peak_memory()
        return peak_mem
    except RuntimeError as e:
        if "out of memory" in str(e):
            return None
        raise e
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

print(f"{'Mode':<10} | {'Batch':<5} | {'Memory (MB)':<10}")
print("-" * 30)

modes = ['standard', 'cggr']
batch_sizes = [1, 2, 4, 8, 16, 32, 64]

results = {}

for mode in modes:
    max_bs = 0
    for bs in batch_sizes:
        try:
            mem = benchmark_memory(bs, mode)
            if mem:
                print(f"{mode:<10} | {bs:<5} | {mem:.1f}")
                max_bs = bs
            else:
                print(f"{mode:<10} | {bs:<5} | OOM")
                break
        except Exception as e:
             print(f"{mode:<10} | {bs:<5} | Error: {e}")
             break
    results[mode] = max_bs

print("\nSummary:")
print(f"Standard Max BS: {results['standard']}")
print(f"CGGR Max BS:     {results['cggr']}")
print(f"Multiplier:      {results['cggr'] / results['standard']:.2f}x")
