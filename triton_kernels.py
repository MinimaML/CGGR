"""
CGGR Triton Kernels - Advanced Version
=======================================
Fused CUDA kernels for Confidence-Gated Gradient Routing.
Includes multi-strategy scoring, dynamic thresholds, and stratified sampling.
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# DIFFICULTY SCORING KERNELS
# =============================================================================

@triton.jit
def _fused_scoring_kernel(
    logits_ptr,
    targets_ptr,
    difficulty_ptr,
    confidence_ptr,
    entropy_ptr,
    vocab_size: tl.constexpr,
    has_targets: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel computing entropy, confidence, margin, and optionally loss-based difficulty.
    Outputs all metrics so caller can combine as needed.
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
    
    # Pass 3: Compute entropy, confidence (max prob), and second-best prob (for margin)
    entropy_acc = 0.0
    top1_prob = 0.0
    top2_prob = 0.0
    
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        vals = tl.load(logits_ptr + row_start + offsets, mask=mask, other=float('-inf'))
        exp_vals = tl.exp(vals - max_val)
        probs = exp_vals / exp_sum
        
        # Entropy
        log_probs = tl.log(probs + 1e-10)
        entropy_acc += tl.sum(-probs * log_probs, axis=0)
        
        # Track top-2 probs for margin
        block_max = tl.max(probs, axis=0)
        if block_max > top1_prob:
            top2_prob = top1_prob
            top1_prob = block_max
        elif block_max > top2_prob:
            top2_prob = block_max
    
    # Confidence = top1_prob
    confidence = top1_prob
    
    # Margin-based difficulty: smaller margin = harder
    margin = top1_prob - top2_prob
    margin_difficulty = 1.0 - margin
    
    # Entropy-based difficulty
    entropy_difficulty = entropy_acc
    
    # Combined difficulty: entropy + margin (normalized)
    # Higher = harder
    difficulty = entropy_difficulty + margin_difficulty
    
    # If we have targets, add loss-based component
    if has_targets:
        target_idx = tl.load(targets_ptr + row_idx)
        target_logit = tl.load(logits_ptr + row_start + target_idx)
        log_sum_exp = tl.log(exp_sum) + max_val
        nll = log_sum_exp - target_logit  # Negative log likelihood
        difficulty = difficulty + nll  # Add loss component
    
    tl.store(difficulty_ptr + row_idx, difficulty)
    tl.store(confidence_ptr + row_idx, confidence)
    tl.store(entropy_ptr + row_idx, entropy_acc)


@triton.jit
def _compute_batch_stats_kernel(
    values_ptr,
    mean_ptr,
    num_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute mean of values for dynamic thresholding."""
    acc = 0.0
    for block_start in range(0, num_tokens, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_tokens
        vals = tl.load(values_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    
    mean = acc / num_tokens
    tl.store(mean_ptr, mean)


@triton.jit
def _stratified_select_kernel(
    difficulty_ptr,
    output_mask_ptr,
    num_tokens: tl.constexpr,
    num_strata: tl.constexpr,
    total_select: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Stratified sampling: divide tokens into difficulty buckets and sample from each.
    Each block handles selection for one stratum.
    """
    stratum_idx = tl.program_id(0)
    
    # Compute stratum boundaries (divide difficulty range into equal parts)
    # Stratum 0 = hardest, Stratum N-1 = easiest
    stratum_lower = stratum_idx / num_strata
    stratum_upper = (stratum_idx + 1) / num_strata
    
    # Tokens per stratum (weighted: harder strata get more)
    # Weight: stratum 0 gets 40%, stratum 1 gets 30%, etc.
    weight = (num_strata - stratum_idx) / ((num_strata * (num_strata + 1)) // 2)
    tokens_for_stratum = tl.floor(total_select * weight).to(tl.int32)
    tokens_for_stratum = tl.maximum(tokens_for_stratum, 1)
    
    # Find tokens in this stratum and mark top ones
    count = 0
    for block_start in range(0, num_tokens, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_tokens
        
        difficulty = tl.load(difficulty_ptr + offsets, mask=mask, other=0.0)
        
        # Normalize difficulty to [0, 1] for stratum assignment
        # Assume difficulty roughly in [-2, 4] range
        normalized = (difficulty + 2.0) / 6.0
        normalized = tl.maximum(0.0, tl.minimum(1.0, normalized))
        
        # Check if in this stratum
        in_stratum = (normalized >= stratum_lower) & (normalized < stratum_upper)
        
        # Mark tokens (simple: mark first N in stratum)
        should_mark = in_stratum & (count < tokens_for_stratum)
        count += tl.sum(should_mark.to(tl.int32), axis=0)
        
        # Atomic OR to set mask bits (1 = selected)
        current_mask = tl.load(output_mask_ptr + offsets, mask=mask, other=0.0)
        new_mask = tl.where(should_mark, 1.0, current_mask)
        tl.store(output_mask_ptr + offsets, new_mask, mask=mask)


@triton.jit
def _sequence_aware_select_kernel(
    difficulty_ptr,
    mask_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    min_per_seq: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Ensure minimum token coverage per sequence.
    Each program handles one sequence.
    """
    seq_idx = tl.program_id(0)
    seq_start = seq_idx * seq_len
    
    # Count currently selected tokens in this sequence
    selected_count = 0
    for block_start in range(0, seq_len, BLOCK_SIZE):
        offsets = seq_start + block_start + tl.arange(0, BLOCK_SIZE)
        mask_valid = (block_start + tl.arange(0, BLOCK_SIZE)) < seq_len
        vals = tl.load(mask_ptr + offsets, mask=mask_valid, other=0.0)
        selected_count += tl.sum((vals > 0.5).to(tl.int32), axis=0)
    
    # If we have enough, done
    if selected_count >= min_per_seq:
        return
    
    # Need to add more tokens - find hardest unselected ones
    need = min_per_seq - selected_count
    
    for block_start in range(0, seq_len, BLOCK_SIZE):
        offsets = seq_start + block_start + tl.arange(0, BLOCK_SIZE)
        pos_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask_valid = pos_offsets < seq_len
        
        difficulty = tl.load(difficulty_ptr + offsets, mask=mask_valid, other=-1000.0)
        current_mask = tl.load(mask_ptr + offsets, mask=mask_valid, other=0.0)
        
        # Find unselected tokens with high difficulty
        is_unselected = current_mask < 0.5
        
        # Mark unselected tokens (simple approach: mark hardest first)
        # For simplicity, just mark first `need` unselected tokens
        should_mark = is_unselected & (need > 0)
        added = tl.sum(should_mark.to(tl.int32), axis=0)
        need = need - added
        
        new_mask = tl.where(should_mark, 1.0, current_mask)
        tl.store(mask_ptr + offsets, new_mask, mask=mask_valid)


@triton.jit
def _topk_select_kernel(
    difficulty_ptr,
    mask_ptr,
    threshold,
    num_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple top-k selection by threshold."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_valid = offsets < num_tokens
    
    difficulty = tl.load(difficulty_ptr + offsets, mask=mask_valid, other=-1000.0)
    
    # Select if difficulty >= threshold
    selected = difficulty >= threshold
    tl.store(mask_ptr + offsets, selected.to(tl.float32), mask=mask_valid)


@triton.jit
def _apply_mask_to_loss_kernel(
    loss_ptr,
    mask_ptr,
    output_ptr,
    num_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply selection mask to per-token loss."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_valid = offsets < num_tokens
    
    loss = tl.load(loss_ptr + offsets, mask=mask_valid, other=0.0)
    selection_mask = tl.load(mask_ptr + offsets, mask=mask_valid, other=0.0)
    
    masked_loss = loss * selection_mask
    tl.store(output_ptr + offsets, masked_loss, mask=mask_valid)


# =============================================================================
# PYTHON WRAPPERS
# =============================================================================

def fused_difficulty_score(
    logits: torch.Tensor,
    targets: torch.Tensor = None,
) -> tuple:
    """
    Compute difficulty scores with optional loss-based component.
    
    Returns:
        difficulty, confidence, entropy
    """
    original_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    
    logits_2d = logits.view(-1, vocab_size).contiguous()
    num_tokens = logits_2d.shape[0]
    
    difficulty = torch.empty(num_tokens, device=logits.device, dtype=logits.dtype)
    confidence = torch.empty(num_tokens, device=logits.device, dtype=logits.dtype)
    entropy = torch.empty(num_tokens, device=logits.device, dtype=logits.dtype)
    
    has_targets = targets is not None
    if has_targets:
        targets_flat = targets.view(-1).contiguous()
    else:
        targets_flat = torch.zeros(num_tokens, device=logits.device, dtype=torch.long)
    
    BLOCK_SIZE = min(1024, triton.next_power_of_2(vocab_size))
    
    _fused_scoring_kernel[(num_tokens,)](
        logits_2d, targets_flat, difficulty, confidence, entropy,
        vocab_size=vocab_size,
        has_targets=has_targets,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (
        difficulty.view(original_shape),
        confidence.view(original_shape),
        entropy.view(original_shape),
    )


def compute_dynamic_threshold(
    confidence: torch.Tensor,
    base_ratio: float,
    sensitivity: float = 0.5,
) -> float:
    """
    Compute dynamic token ratio based on batch confidence.
    Low confidence → more tokens, High confidence → fewer tokens.
    """
    mean_conf = confidence.mean().item()
    # Adjust ratio: if mean_conf is low, increase ratio
    # ratio = base_ratio * (1 + (1 - mean_conf) * sensitivity)
    adjusted_ratio = base_ratio * (1.0 + (1.0 - mean_conf) * sensitivity)
    return min(1.0, max(base_ratio * 0.5, adjusted_ratio))


def select_tokens_stratified(
    difficulty: torch.Tensor,
    total_ratio: float,
    num_strata: int = 4,
) -> torch.Tensor:
    """
    Stratified token selection across difficulty buckets.
    """
    flat_diff = difficulty.view(-1).contiguous()
    num_tokens = flat_diff.numel()
    total_select = int(num_tokens * total_ratio)
    
    mask = torch.zeros(num_tokens, device=difficulty.device, dtype=difficulty.dtype)
    
    BLOCK_SIZE = 1024
    
    _stratified_select_kernel[(num_strata,)](
        flat_diff, mask,
        num_tokens=num_tokens,
        num_strata=num_strata,
        total_select=total_select,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return mask.view(difficulty.shape)


def select_tokens_topk(
    difficulty: torch.Tensor,
    ratio: float,
) -> torch.Tensor:
    """Top-k hardest tokens selection."""
    flat_diff = difficulty.view(-1).contiguous()
    num_tokens = flat_diff.numel()
    k = max(1, int(num_tokens * ratio))
    
    # Find threshold for top-k
    threshold = torch.topk(flat_diff, k).values[-1].item()
    
    mask = torch.zeros(num_tokens, device=difficulty.device, dtype=difficulty.dtype)
    
    BLOCK_SIZE = 1024
    grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    _topk_select_kernel[grid](
        flat_diff, mask, threshold,
        num_tokens=num_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return mask.view(difficulty.shape)


def ensure_sequence_coverage(
    difficulty: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
    min_per_seq: int = 1,
) -> torch.Tensor:
    """Ensure minimum token coverage per sequence."""
    flat_diff = difficulty.view(-1).contiguous()
    flat_mask = mask.view(-1).contiguous().clone()
    
    BLOCK_SIZE = min(1024, triton.next_power_of_2(seq_len))
    
    _sequence_aware_select_kernel[(batch_size,)](
        flat_diff, flat_mask,
        batch_size=batch_size,
        seq_len=seq_len,
        min_per_seq=min_per_seq,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return flat_mask.view(mask.shape)


def apply_mask_to_loss(
    per_token_loss: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply selection mask to loss using Triton kernel."""
    flat_loss = per_token_loss.view(-1).contiguous()
    flat_mask = mask.view(-1).contiguous()
    num_tokens = flat_loss.numel()
    
    output = torch.empty_like(flat_loss)
    
    BLOCK_SIZE = 1024
    grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    _apply_mask_to_loss_kernel[grid](
        flat_loss, flat_mask, output,
        num_tokens=num_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.view(per_token_loss.shape)


# Legacy exports for backward compatibility
def triton_fused_difficulty_score(logits):
    return fused_difficulty_score(logits, targets=None)

def triton_bucket_assign(difficulty, num_buckets, num_layers, warmup_progress):
    # Legacy function - now just returns token selection mask
    ratio = 1.0 - warmup_progress * 0.75  # 100% -> 25%
    return select_tokens_topk(difficulty, ratio)


class TritonGradientMask(torch.autograd.Function):
    """Legacy - kept for backward compatibility but deprecated."""
    
    @staticmethod
    def forward(ctx, tensor, stop_layers, layer_idx, leak_rate):
        return tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None
