"""
Benchmark: Persistent CGGR Kernels vs Standard CGGR
====================================================
Compares performance of persistent kernel execution against standard
kernel-per-operation approach.

Expected improvement: 10-15% faster end-to-end throughput
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persistent_cggr_kernels import (
    PersistentTRMLP,
    PersistentKernelConfig,
    create_persistent_tr_mlp,
    HAS_TRITON,
)


class StandardMoE(nn.Module):
    """Standard Mixture of Experts implementation for comparison."""
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        
        # Expert MLPs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim, bias=False),
                nn.SiLU(),
                nn.Linear(intermediate_dim, hidden_dim, bias=False),
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        if x.dim() == 3:
            batch, seq, hidden = x.shape
            x = x.view(-1, hidden)
        
        # Router
        router_logits = self.router(x)
        routing_weights = F.softmax(router_logits, dim=-1)
        routing_weights, expert_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Standard expert execution
        output = torch.zeros_like(x)
        
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = (expert_indices[:, k] == e)
                if not mask.any():
                    continue
                
                token_indices = mask.nonzero(as_tuple=True)[0]
                expert_input = x[token_indices]
                expert_weights = routing_weights[token_indices, k:k+1]
                
                expert_output = self.experts[e](expert_input)
                output[token_indices] += expert_output * expert_weights
        
        # Aux loss
        num_tokens = x.shape[0]
        expert_mask = F.one_hot(expert_indices[:, 0], self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=0)
        router_prob = F.softmax(router_logits, dim=-1).mean(dim=0)
        aux_loss = (tokens_per_expert / num_tokens * router_prob).sum() * self.num_experts
        
        if len(original_shape) == 3:
            output = output.view(original_shape)
        
        return output, aux_loss


def benchmark_forward(
    model: nn.Module,
    x: torch.Tensor,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> Dict[str, float]:
    """Benchmark forward pass timing."""
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(x)
    
    torch.cuda.synchronize()
    
    # Timed iterations
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(num_iterations):
        start_event.record()
        with torch.no_grad():
            output, _ = model(x)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    
    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def benchmark_backward(
    model: nn.Module,
    x: torch.Tensor,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> Dict[str, float]:
    """Benchmark forward + backward pass timing."""
    # Warmup
    for _ in range(num_warmup):
        x_grad = x.clone().requires_grad_(True)
        output, aux_loss = model(x_grad)
        loss = output.sum() + aux_loss
        loss.backward()
    
    torch.cuda.synchronize()
    
    # Timed iterations
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(num_iterations):
        x_grad = x.clone().requires_grad_(True)
        
        start_event.record()
        output, aux_loss = model(x_grad)
        loss = output.sum() + aux_loss
        loss.backward()
        end_event.record()
        
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    
    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def run_benchmark(
    batch_size: int = 4,
    seq_len: int = 512,
    hidden_dim: int = 1024,
    intermediate_dim: int = 4096,
    num_experts: int = 8,
    top_k: int = 2,
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = 'cuda',
) -> Dict[str, Dict[str, float]]:
    """Run full benchmark comparison."""
    
    print(f"\n{'='*60}")
    print(f"Persistent CGGR Kernels Benchmark")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Batch Size:       {batch_size}")
    print(f"  Sequence Length:  {seq_len}")
    print(f"  Hidden Dim:       {hidden_dim}")
    print(f"  Intermediate Dim: {intermediate_dim}")
    print(f"  Num Experts:      {num_experts}")
    print(f"  Top-K:            {top_k}")
    print(f"  Device:           {device}")
    print(f"  Triton Available: {HAS_TRITON}")
    print(f"{'='*60}\n")
    
    # Create models
    standard_moe = StandardMoE(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        top_k=top_k,
    ).to(device).half()
    
    persistent_moe = create_persistent_tr_mlp(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        top_k=top_k,
    ).to(device).half()
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float16)
    
    results = {}
    
    # Benchmark Standard MoE
    print("Benchmarking Standard MoE...")
    standard_fwd = benchmark_forward(standard_moe, x, num_warmup, num_iterations)
    standard_bwd = benchmark_backward(standard_moe, x, num_warmup, num_iterations)
    results['standard'] = {
        'forward': standard_fwd,
        'backward': standard_bwd,
    }
    print(f"  Forward:  {standard_fwd['mean_ms']:.3f} ms (±{standard_fwd['std_ms']:.3f})")
    print(f"  Backward: {standard_bwd['mean_ms']:.3f} ms (±{standard_bwd['std_ms']:.3f})")
    
    # Benchmark Persistent CGGR MoE
    print("\nBenchmarking Persistent CGGR MoE...")
    persistent_fwd = benchmark_forward(persistent_moe, x, num_warmup, num_iterations)
    persistent_bwd = benchmark_backward(persistent_moe, x, num_warmup, num_iterations)
    results['persistent'] = {
        'forward': persistent_fwd,
        'backward': persistent_bwd,
    }
    print(f"  Forward:  {persistent_fwd['mean_ms']:.3f} ms (±{persistent_fwd['std_ms']:.3f})")
    print(f"  Backward: {persistent_bwd['mean_ms']:.3f} ms (±{persistent_bwd['std_ms']:.3f})")
    
    # Calculate speedup
    fwd_speedup = standard_fwd['mean_ms'] / persistent_fwd['mean_ms']
    bwd_speedup = standard_bwd['mean_ms'] / persistent_bwd['mean_ms']
    total_speedup = (standard_fwd['mean_ms'] + standard_bwd['mean_ms']) / \
                   (persistent_fwd['mean_ms'] + persistent_bwd['mean_ms'])
    
    print(f"\n{'='*60}")
    print("Performance Comparison:")
    print(f"{'='*60}")
    print(f"  Forward Speedup:  {fwd_speedup:.2f}x")
    print(f"  Backward Speedup: {bwd_speedup:.2f}x")
    print(f"  Total Speedup:    {total_speedup:.2f}x ({(total_speedup-1)*100:.1f}% faster)")
    print(f"{'='*60}\n")
    
    results['speedup'] = {
        'forward': fwd_speedup,
        'backward': bwd_speedup,
        'total': total_speedup,
    }
    
    return results


def run_scaling_benchmark(
    device: str = 'cuda',
) -> None:
    """Run benchmark across multiple configurations."""
    
    print("\n" + "="*70)
    print("SCALING BENCHMARK: Persistent CGGR vs Standard MoE")
    print("="*70 + "\n")
    
    configs = [
        # (batch, seq, hidden, intermediate, experts, topk)
        (1, 512, 768, 3072, 4, 1),
        (1, 512, 768, 3072, 8, 2),
        (4, 512, 1024, 4096, 8, 2),
        (4, 1024, 1024, 4096, 8, 2),
        (8, 512, 2048, 8192, 8, 2),
        (4, 512, 2048, 8192, 16, 2),
    ]
    
    print(f"{'Config':<40} | {'Standard':<12} | {'Persistent':<12} | {'Speedup':<10}")
    print("-"*80)
    
    for batch, seq, hidden, inter, experts, topk in configs:
        config_str = f"B={batch}, S={seq}, H={hidden}, E={experts}"
        
        try:
            results = run_benchmark(
                batch_size=batch,
                seq_len=seq,
                hidden_dim=hidden,
                intermediate_dim=inter,
                num_experts=experts,
                top_k=topk,
                num_warmup=5,
                num_iterations=20,
                device=device,
            )
            
            std_total = results['standard']['forward']['mean_ms'] + results['standard']['backward']['mean_ms']
            pers_total = results['persistent']['forward']['mean_ms'] + results['persistent']['backward']['mean_ms']
            speedup = results['speedup']['total']
            
            print(f"{config_str:<40} | {std_total:>10.2f}ms | {pers_total:>10.2f}ms | {speedup:>8.2f}x")
            
        except Exception as e:
            print(f"{config_str:<40} | {'ERROR':<12} | {str(e)[:20]}")
        
        # Clear cache between configs
        torch.cuda.empty_cache()
    
    print("\n" + "="*70 + "\n")


def verify_correctness(
    hidden_dim: int = 512,
    device: str = 'cuda',
) -> bool:
    """Verify that persistent kernels produce correct results."""
    
    print("Verifying numerical correctness...")
    
    torch.manual_seed(42)
    
    batch, seq = 2, 64
    intermediate_dim = hidden_dim * 4
    num_experts = 4
    top_k = 2
    
    # Create models with same initialization
    persistent_model = create_persistent_tr_mlp(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        top_k=top_k,
    ).to(device).float()  # Use float32 for better precision comparison
    
    # Create input
    x = torch.randn(batch, seq, hidden_dim, device=device)
    
    # Forward pass with persistent kernels
    persistent_model.eval()
    with torch.no_grad():
        output_persistent, _ = persistent_model(x, use_persistent=True)
    
    # Forward pass with PyTorch fallback
    with torch.no_grad():
        output_pytorch, _ = persistent_model(x, use_persistent=False)
    
    # Compare outputs
    diff = (output_persistent - output_pytorch).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    rtol, atol = 1e-3, 1e-4
    is_close = torch.allclose(output_persistent, output_pytorch, rtol=rtol, atol=atol)
    
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    print(f"  Results match (rtol={rtol}, atol={atol}): {is_close}")
    
    return is_close


def main():
    parser = argparse.ArgumentParser(description="Benchmark Persistent CGGR Kernels")
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length')
    parser.add_argument('--hidden-dim', type=int, default=1024, help='Hidden dimension')
    parser.add_argument('--intermediate-dim', type=int, default=4096, help='Intermediate dimension')
    parser.add_argument('--num-experts', type=int, default=8, help='Number of experts')
    parser.add_argument('--top-k', type=int, default=2, help='Top-K routing')
    parser.add_argument('--num-warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--num-iterations', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--scaling', action='store_true', help='Run scaling benchmark')
    parser.add_argument('--verify', action='store_true', help='Verify correctness')
    parser.add_argument('--compare', action='store_true', help='Run comparison (default)')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA not available. Benchmarks require GPU.")
        return
    
    device = 'cuda'
    
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print(f"Triton Available: {HAS_TRITON}")
    
    if args.verify:
        verify_correctness(device=device)
    
    if args.scaling:
        run_scaling_benchmark(device=device)
    elif args.compare or not (args.verify or args.scaling):
        run_benchmark(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            hidden_dim=args.hidden_dim,
            intermediate_dim=args.intermediate_dim,
            num_experts=args.num_experts,
            top_k=args.top_k,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
            device=device,
        )


if __name__ == '__main__':
    main()
