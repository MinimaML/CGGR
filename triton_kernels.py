"""
CGGR Triton Kernels
===================
Fused CUDA kernels for Confidence-Gated Gradient Routing.
Triton is required for this library to function.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_difficulty_kernel(
    logits_ptr,
    difficulty_ptr,
    confidence_ptr,
    entropy_ptr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: softmax -> max (confidence) -> entropy -> difficulty
    Each program handles one token (one row of logits).
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * vocab_size
    
    # Pass 1: Find max for numerical stability
    max_val = float('-inf')
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        vals = tl.load(logits_ptr + row_start + offsets, mask=mask, other=float('-inf'))
        max_val = tl.maximum(max_val, tl.max(vals, axis=0))
    
    # Pass 2: Compute exp sum
    exp_sum = 0.0
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        vals = tl.load(logits_ptr + row_start + offsets, mask=mask, other=float('-inf'))
        exp_vals = tl.exp(vals - max_val)
        exp_sum += tl.sum(exp_vals, axis=0)
    
    # Pass 3: Compute entropy and confidence
    entropy_acc = 0.0
    max_prob = 0.0
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        vals = tl.load(logits_ptr + row_start + offsets, mask=mask, other=float('-inf'))
        exp_vals = tl.exp(vals - max_val)
        probs = exp_vals / exp_sum
        
        log_probs = tl.log(probs + 1e-10)
        entropy_acc += tl.sum(-probs * log_probs, axis=0)
        max_prob = tl.maximum(max_prob, tl.max(probs, axis=0))
    
    # Difficulty = Entropy - Confidence
    tl.store(difficulty_ptr + row_idx, entropy_acc - max_prob)
    tl.store(confidence_ptr + row_idx, max_prob)
    tl.store(entropy_ptr + row_idx, entropy_acc)


@triton.jit
def _bucket_assign_kernel(
    difficulty_ptr,
    stop_layers_ptr,
    num_tokens: tl.constexpr,
    num_buckets: tl.constexpr,
    num_layers: tl.constexpr,
    warmup_progress: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Assign tokens to buckets based on difficulty percentile.
    O(n) complexity using histogram-based approach.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_tokens
    
    difficulties = tl.load(difficulty_ptr + offsets, mask=mask, other=0.0)
    
    # Normalize to [0, 1] (assume difficulty in [-2, 2])
    normalized = (difficulties + 2.0) / 4.0
    normalized = tl.maximum(0.0, tl.minimum(1.0, normalized))
    
    # Higher difficulty = lower bucket = more layers
    bucket_float = (1.0 - normalized) * num_buckets
    bucket_idx = tl.minimum(tl.floor(bucket_float).to(tl.int32), num_buckets - 1)
    
    # Apply curriculum
    if warmup_progress < 1.0:
        allowed_max = tl.floor(warmup_progress * (num_buckets - 1)).to(tl.int32)
        bucket_idx = tl.minimum(bucket_idx, allowed_max)
    
    # Convert bucket to stop layer
    chunk_size = num_layers // num_buckets
    layers_to_keep = (num_buckets - bucket_idx) * chunk_size
    stop_layer = num_layers - layers_to_keep - 1
    
    tl.store(stop_layers_ptr + offsets, stop_layer, mask=mask)


@triton.jit
def _gradient_mask_kernel(
    grad_ptr,
    stop_layers_ptr,
    out_ptr,
    layer_idx: tl.constexpr,
    leak_rate: tl.constexpr,
    numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply gradient mask based on stop layers."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    stop_layer = tl.load(stop_layers_ptr + offsets, mask=mask, other=-1)
    
    should_pass = stop_layer < layer_idx
    
    if leak_rate > 0.0:
        grad_mask = tl.where(should_pass, 1.0, leak_rate)
    else:
        grad_mask = tl.where(should_pass, 1.0, 0.0)
    
    tl.store(out_ptr + offsets, grad * grad_mask, mask=mask)


def triton_fused_difficulty_score(logits: torch.Tensor):
    """Compute difficulty, confidence, entropy in one fused kernel."""
    original_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    
    logits_2d = logits.view(-1, vocab_size).contiguous()
    num_tokens = logits_2d.shape[0]
    
    difficulty = torch.empty(num_tokens, device=logits.device, dtype=logits.dtype)
    confidence = torch.empty(num_tokens, device=logits.device, dtype=logits.dtype)
    entropy = torch.empty(num_tokens, device=logits.device, dtype=logits.dtype)
    
    BLOCK_SIZE = min(1024, triton.next_power_of_2(vocab_size))
    
    _fused_difficulty_kernel[(num_tokens,)](
        logits_2d, difficulty, confidence, entropy,
        vocab_size=vocab_size, BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (
        difficulty.view(original_shape),
        confidence.view(original_shape),
        entropy.view(original_shape),
    )


def triton_bucket_assign(
    difficulty: torch.Tensor,
    num_buckets: int,
    num_layers: int,
    warmup_progress: float,
) -> torch.Tensor:
    """Assign stop layers to tokens based on difficulty."""
    original_shape = difficulty.shape
    flat_diff = difficulty.view(-1).contiguous()
    num_tokens = flat_diff.numel()
    
    stop_layers = torch.empty(num_tokens, device=difficulty.device, dtype=torch.int32)
    
    BLOCK_SIZE = 1024
    grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    _bucket_assign_kernel[grid](
        flat_diff, stop_layers,
        num_tokens=num_tokens,
        num_buckets=num_buckets,
        num_layers=num_layers,
        warmup_progress=warmup_progress,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return stop_layers.view(original_shape)


class TritonGradientMask(torch.autograd.Function):
    """Custom autograd function for Triton gradient masking."""
    
    @staticmethod
    def forward(ctx, tensor, stop_layers, layer_idx, leak_rate):
        ctx.save_for_backward(stop_layers)
        ctx.layer_idx = layer_idx
        ctx.leak_rate = leak_rate
        return tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        stop_layers, = ctx.saved_tensors
        layer_idx = ctx.layer_idx
        leak_rate = ctx.leak_rate
        
        grad_flat = grad_output.contiguous().view(-1)
        stop_flat = stop_layers.view(-1).contiguous()
        
        # Expand stop_layers to match grad dimensions
        hidden_dim = grad_output.shape[-1] if grad_output.dim() > stop_layers.dim() else 1
        if hidden_dim > 1:
            stop_flat = stop_flat.unsqueeze(-1).expand(-1, hidden_dim).contiguous().view(-1)
        
        out = torch.empty_like(grad_flat)
        numel = grad_flat.numel()
        
        BLOCK_SIZE = 1024
        grid = ((numel + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        
        _gradient_mask_kernel[grid](
            grad_flat, stop_flat, out,
            layer_idx=layer_idx,
            leak_rate=leak_rate,
            numel=numel,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return out.view(grad_output.shape), None, None, None
