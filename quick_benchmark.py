"""Quick benchmark for persistent CGGR kernels."""
import torch
import time

from persistent_cggr_kernels import create_persistent_tr_mlp

def benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"}')
    
    # Create models
    model = create_persistent_tr_mlp(
        hidden_dim=512,
        intermediate_dim=2048,
        num_experts=8,
        top_k=2,
    ).to(device).half()
    
    x = torch.randn(4, 128, 512, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(x, use_persistent=False)
    
    torch.cuda.synchronize()
    
    # Benchmark forward
    num_iters = 50
    start = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            output, _ = model(x, use_persistent=False)
    torch.cuda.synchronize()
    fwd_time = (time.perf_counter() - start) / num_iters * 1000
    
    # Benchmark forward + backward
    start = time.perf_counter()
    for _ in range(num_iters):
        x_grad = x.clone().requires_grad_(True)
        output, aux = model(x_grad, use_persistent=False)
        loss = output.sum() + aux
        loss.backward()
    torch.cuda.synchronize()
    fwd_bwd_time = (time.perf_counter() - start) / num_iters * 1000
    
    print(f'\nPersistent CGGR MoE (PyTorch mode):')
    print(f'  Forward:  {fwd_time:.2f} ms')
    print(f'  Fwd+Bwd:  {fwd_bwd_time:.2f} ms')
    tokens = 4 * 128
    print(f'\nTokens processed: {tokens} per batch')
    print(f'Throughput: {tokens / (fwd_time / 1000):.0f} tokens/sec (inference)')
    print(f'\nBenchmark complete!')

if __name__ == '__main__':
    benchmark()

