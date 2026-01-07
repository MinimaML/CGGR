"""
CGGR Triton Kernels
===================
Fused CUDA kernels for Confidence-Gated Gradient Routing.
Provides significant speedup over PyTorch operations when Triton is available.
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


if HAS_TRITON:
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
        
        # Pointer to this row's logits
        row_start = row_idx * vocab_size
        
        # --- Pass 1: Find max for numerical stability ---
        max_val = float('-inf')
        for block_start in range(0, vocab_size, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < vocab_size
            vals = tl.load(logits_ptr + row_start + offsets, mask=mask, other=float('-inf'))
            max_val = tl.maximum(max_val, tl.max(vals, axis=0))
        
        # --- Pass 2: Compute exp sum and find max prob (confidence) ---
        exp_sum = 0.0
        max_prob = 0.0
        
        for block_start in range(0, vocab_size, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < vocab_size
            vals = tl.load(logits_ptr + row_start + offsets, mask=mask, other=float('-inf'))
            exp_vals = tl.exp(vals - max_val)
            exp_sum += tl.sum(exp_vals, axis=0)
        
        # --- Pass 3: Compute entropy and confidence ---
        entropy_acc = 0.0
        max_prob = 0.0
        
        for block_start in range(0, vocab_size, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < vocab_size
            vals = tl.load(logits_ptr + row_start + offsets, mask=mask, other=float('-inf'))
            exp_vals = tl.exp(vals - max_val)
            probs = exp_vals / exp_sum
            
            # Entropy: -sum(p * log(p)), handle p=0 case
            log_probs = tl.log(probs + 1e-10)
            entropy_acc += tl.sum(-probs * log_probs, axis=0)
            
            # Track max prob (confidence)
            max_prob = tl.maximum(max_prob, tl.max(probs, axis=0))
        
        # Difficulty = Entropy - Confidence (higher = harder)
        difficulty = entropy_acc - max_prob
        
        # Store results
        tl.store(difficulty_ptr + row_idx, difficulty)
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
        Uses histogram-based approach for O(n) complexity.
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_tokens
        
        # Load difficulties for this block
        difficulties = tl.load(difficulty_ptr + offsets, mask=mask, other=0.0)
        
        # Normalize to [0, 1] range for bucketing
        # We use a simple approach: assume difficulties are roughly in [-2, 2] range
        normalized = (difficulties + 2.0) / 4.0
        normalized = tl.maximum(0.0, tl.minimum(1.0, normalized))
        
        # Assign to bucket (higher difficulty = lower bucket = more layers)
        # Bucket 0 = hardest (all layers), Bucket N-1 = easiest (fewest layers)
        bucket_float = (1.0 - normalized) * num_buckets
        bucket_idx = tl.minimum(tl.floor(bucket_float).to(tl.int32), num_buckets - 1)
        
        # Apply curriculum: limit max bucket based on warmup progress
        if warmup_progress < 1.0:
            allowed_max = tl.floor(warmup_progress * (num_buckets - 1)).to(tl.int32)
            bucket_idx = tl.minimum(bucket_idx, allowed_max)
        
        # Convert bucket to stop layer
        # Bucket 0 -> stop at layer -1 (never stop = full gradient)
        # Bucket N-1 -> stop at layer (num_layers - num_layers/N)
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
        """
        Apply gradient mask based on stop layers.
        Gradients pass if stop_layer < layer_idx, otherwise masked (or leaked).
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
        stop_layer = tl.load(stop_layers_ptr + offsets, mask=mask, other=-1)
        
        # Pass if stop_layer < layer_idx
        should_pass = stop_layer < layer_idx
        
        if leak_rate > 0.0:
            # Mask = should_pass * (1 - leak_rate) + leak_rate
            grad_mask = tl.where(should_pass, 1.0, leak_rate)
        else:
            grad_mask = tl.where(should_pass, 1.0, 0.0)
        
        out = grad * grad_mask
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_difficulty_score(logits: torch.Tensor):
    """
    Compute difficulty, confidence, and entropy in a single fused kernel.
    
    Args:
        logits: (batch_size, seq_len, vocab_size) or (num_tokens, vocab_size)
        
    Returns:
        difficulty, confidence, entropy - each (batch_size, seq_len) or (num_tokens,)
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available")
    
    original_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    
    # Flatten to 2D
    logits_2d = logits.view(-1, vocab_size).contiguous()
    num_tokens = logits_2d.shape[0]
    
    # Allocate outputs
    difficulty = torch.empty(num_tokens, device=logits.device, dtype=logits.dtype)
    confidence = torch.empty(num_tokens, device=logits.device, dtype=logits.dtype)
    entropy = torch.empty(num_tokens, device=logits.device, dtype=logits.dtype)
    
    # Launch kernel - one program per token
    BLOCK_SIZE = min(1024, triton.next_power_of_2(vocab_size))
    grid = (num_tokens,)
    
    _fused_difficulty_kernel[grid](
        logits_2d,
        difficulty,
        confidence,
        entropy,
        vocab_size=vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return difficulty.view(original_shape), confidence.view(original_shape), entropy.view(original_shape)


def triton_bucket_assign(
    difficulty: torch.Tensor,
    num_buckets: int,
    num_layers: int,
    warmup_progress: float,
) -> torch.Tensor:
    """
    Assign stop layers to tokens based on difficulty using Triton kernel.
    
    Args:
        difficulty: (batch_size, seq_len) difficulty scores
        num_buckets: Number of gradient depth buckets
        num_layers: Total number of model layers
        warmup_progress: Curriculum progress [0, 1]
        
    Returns:
        stop_layers: (batch_size, seq_len) integer stop layer indices
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available")
    
    original_shape = difficulty.shape
    flat_diff = difficulty.view(-1).contiguous()
    num_tokens = flat_diff.numel()
    
    stop_layers = torch.empty(num_tokens, device=difficulty.device, dtype=torch.int32)
    
    BLOCK_SIZE = 1024
    grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    _bucket_assign_kernel[grid](
        flat_diff,
        stop_layers,
        num_tokens=num_tokens,
        num_buckets=num_buckets,
        num_layers=num_layers,
        warmup_progress=warmup_progress,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return stop_layers.view(original_shape)


class TritonGradientMask(torch.autograd.Function):
    """
    Custom autograd function that applies gradient masking with Triton.
    """
    
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
        
        if not HAS_TRITON or not grad_output.is_cuda:
            # Fallback to PyTorch
            stop_expanded = stop_layers
            if stop_expanded.dim() < grad_output.dim():
                stop_expanded = stop_expanded.unsqueeze(-1)
            should_pass = (stop_expanded < layer_idx).float()
            if leak_rate > 0:
                mask = should_pass * (1.0 - leak_rate) + leak_rate
            else:
                mask = should_pass
            return grad_output * mask, None, None, None
        
        # Use Triton kernel
        grad_flat = grad_output.contiguous().view(-1)
        stop_flat = stop_layers.view(-1).contiguous()
        
        # Handle shape mismatch (stop_layers is per-token, grad is per-token-per-dim)
        hidden_dim = grad_output.shape[-1] if grad_output.dim() > stop_layers.dim() else 1
        if hidden_dim > 1:
            stop_flat = stop_flat.unsqueeze(-1).expand(-1, hidden_dim).contiguous().view(-1)
        
        out = torch.empty_like(grad_flat)
        numel = grad_flat.numel()
        
        BLOCK_SIZE = 1024
        grid = ((numel + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        
        _gradient_mask_kernel[grid](
            grad_flat,
            stop_flat,
            out,
            layer_idx=layer_idx,
            leak_rate=leak_rate,
            numel=numel,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return out.view(grad_output.shape), None, None, None


def apply_triton_gradient_mask(tensor, stop_layers, layer_idx, leak_rate=0.0):
    """Wrapper to apply Triton gradient masking."""
    return TritonGradientMask.apply(tensor, stop_layers, layer_idx, leak_rate)
