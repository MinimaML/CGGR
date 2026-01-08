"""
CGGR Triton Kernels - With PyTorch Fallback
============================================
Fused CUDA kernels for Confidence-Gated Gradient Routing.
Falls back to PyTorch ops when Triton isn't available.
"""

import torch
import torch.nn.functional as F

# Try to import Triton
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


# =============================================================================
# TRITON KERNELS (only defined if Triton available)
# =============================================================================

if HAS_TRITON:
    @triton.jit
    def _fused_scoring_kernel(
        logits_ptr,
        targets_ptr,
        difficulty_ptr,
        confidence_ptr,
        entropy_ptr,
        vocab_size: tl.constexpr,
        has_targets: tl.constexpr,
        scoring_mode: tl.constexpr,  # 0: entropy, 1: margin, 2: loss, 3: combined
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused kernel for entropy, confidence, and difficulty scoring."""
        row_idx = tl.program_id(0)
        row_start = row_idx * vocab_size
        
        # Pass 1: Find max for stability
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
        
        # Pass 3: Compute entropy, top1, and top2
        entropy_acc = 0.0
        top1_acc = 0.0
        top2_acc = 0.0
        
        for block_start in range(0, vocab_size, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < vocab_size
            vals = tl.load(logits_ptr + row_start + offsets, mask=mask, other=float('-inf'))
            exp_vals = tl.exp(vals - max_val)
            probs = exp_vals / exp_sum
            
            log_probs = tl.log(probs + 1e-10)
            entropy_acc += tl.sum(-probs * log_probs, axis=0)
            
            # Vectorized top-1/2 approx (coarse because it's across blocks)
            # We'll just track global m1, m2 across loops
            block_m1 = tl.max(probs, axis=0)
            if block_m1 > top1_acc:
                top2_acc = tl.maximum(top1_acc, tl.max(tl.where(probs == block_m1, 0.0, probs), axis=0))
                top1_acc = block_m1
            else:
                top2_acc = tl.maximum(top2_acc, block_m1)
        
        # Final difficulty based on mode
        diff = 0.0
        if scoring_mode == 0:  # entropy
            diff = entropy_acc - top1_acc
        elif scoring_mode == 1:  # margin
            diff = 1.0 - (top1_acc - top2_acc)
        elif scoring_mode == 2:  # loss (placeholder, NLL added below if has_targets)
            diff = entropy_acc
        else:  # combined
            diff = (entropy_acc - top1_acc) + (1.0 - (top1_acc - top2_acc))
            
        if has_targets:
            target_idx = tl.load(targets_ptr + row_idx)
            target_logit = tl.load(logits_ptr + row_start + target_idx)
            log_sum_exp = tl.log(exp_sum) + max_val
            nll = log_sum_exp - target_logit
            if scoring_mode == 2: # loss mode
                diff = nll
            else: # add nll to other modes
                diff = diff + nll
        
        tl.store(difficulty_ptr + row_idx, diff)
        tl.store(confidence_ptr + row_idx, top1_acc)
        tl.store(entropy_ptr + row_idx, entropy_acc)


# =============================================================================
# PYTORCH FALLBACK IMPLEMENTATIONS
# =============================================================================

def _pytorch_difficulty_score(logits: torch.Tensor, targets: torch.Tensor = None):
    """PyTorch implementation of difficulty scoring."""
    probs = F.softmax(logits, dim=-1)
    confidence, _ = torch.max(probs, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    difficulty = entropy - confidence
    
    if targets is not None:
        # Add NLL component
        nll = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            targets.view(-1),
            reduction='none'
        ).view(logits.shape[:-1])
        difficulty = difficulty + nll
    
    return difficulty, confidence, entropy


def _pytorch_select_topk(difficulty: torch.Tensor, ratio: float) -> torch.Tensor:
    """PyTorch top-k selection."""
    flat = difficulty.view(-1)
    k = max(1, int(flat.numel() * ratio))
    _, indices = torch.topk(flat, k)
    mask = torch.zeros_like(flat)
    mask[indices] = 1.0
    return mask.view(difficulty.shape)


def _pytorch_stratified_select(
    difficulty: torch.Tensor, 
    total_ratio: float, 
    num_strata: int = 4
) -> torch.Tensor:
    """PyTorch stratified selection."""
    flat = difficulty.view(-1)
    num_tokens = flat.numel()
    total_select = int(num_tokens * total_ratio)
    
    # Sort by difficulty
    sorted_idx = torch.argsort(flat, descending=True)
    
    # Divide into strata and sample from each
    mask = torch.zeros_like(flat)
    tokens_per_stratum = total_select // num_strata
    stratum_size = num_tokens // num_strata
    
    for i in range(num_strata):
        start = i * stratum_size
        end = start + stratum_size if i < num_strata - 1 else num_tokens
        stratum_indices = sorted_idx[start:end]
        
        # Take top tokens from each stratum (more from harder strata)
        weight = (num_strata - i) / sum(range(1, num_strata + 1))
        n_select = max(1, int(total_select * weight))
        select_indices = stratum_indices[:n_select]
        mask[select_indices] = 1.0
    
    return mask.view(difficulty.shape)


# =============================================================================
# UNIFIED API (auto-selects Triton or PyTorch)
# =============================================================================

def fused_difficulty_score(
    logits: torch.Tensor,
    targets: torch.Tensor = None,
    mode: str = 'combined',
) -> tuple:
    """Compute difficulty scores. Uses Triton if available, else PyTorch."""
    
    if HAS_TRITON and logits.is_cuda:
        try:
            return _triton_difficulty_score(logits, targets, mode)
        except Exception:
            # Fallback on any Triton error
            pass
    
    return _pytorch_difficulty_score(logits, targets)


def _triton_difficulty_score(logits: torch.Tensor, targets: torch.Tensor = None, mode: str = 'combined'):
    """Triton implementation wrapper."""
    original_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    
    logits_2d = logits.view(-1, vocab_size).contiguous()
    num_tokens = logits_2d.shape[0]
    
    difficulty = torch.empty(num_tokens, device=logits.device, dtype=logits.dtype)
    confidence = torch.empty(num_tokens, device=logits.device, dtype=logits.dtype)
    entropy = torch.empty(num_tokens, device=logits.device, dtype=logits.dtype)
    
    has_targets = targets is not None
    targets_flat = targets.view(-1).contiguous() if has_targets else torch.zeros(
        num_tokens, device=logits.device, dtype=torch.long
    )
    
    # Map mode string to int
    mode_map = {'entropy': 0, 'margin': 1, 'loss': 2, 'combined': 3}
    mode_int = mode_map.get(mode, 3)
    
    BLOCK_SIZE = min(1024, triton.next_power_of_2(vocab_size))
    
    _fused_scoring_kernel[(num_tokens,)](
        logits_2d, targets_flat, difficulty, confidence, entropy,
        vocab_size=vocab_size,
        has_targets=has_targets,
        scoring_mode=mode_int,
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
    """Compute dynamic token ratio based on batch confidence."""
    mean_conf = confidence.mean().item()
    adjusted_ratio = base_ratio * (1.0 + (1.0 - mean_conf) * sensitivity)
    return min(1.0, max(base_ratio * 0.5, adjusted_ratio))


@triton.jit
def _mask_threshold_kernel(
    input_ptr,
    mask_ptr,
    threshold,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(input_ptr + offsets, mask=mask)
    tl.store(mask_ptr + offsets, (vals >= threshold).to(tl.float32), mask=mask)

def select_tokens_topk(difficulty: torch.Tensor, ratio: float) -> torch.Tensor:
    """Top-k hardest tokens selection. Optimized with Triton for large N."""
    if not HAS_TRITON or not difficulty.is_cuda:
        return _pytorch_select_topk(difficulty, ratio)
    
    try:
        num_tokens = difficulty.numel()
        k = max(1, int(num_tokens * ratio))
        
        # For top-k threshold, we still use torch.kthvalue or torch.topk 
        # as it's highly optimized and hard to beat with a custom Triton histogram for general N.
        # But we use Triton for the final mask construction which is memory-bound.
        threshold = torch.topk(difficulty.view(-1), k).values[-1]
        
        mask = torch.empty_like(difficulty)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)
        
        _mask_threshold_kernel[grid](
            difficulty, mask, threshold, num_tokens, BLOCK_SIZE=BLOCK_SIZE
        )
        return mask
    except Exception:
        return _pytorch_select_topk(difficulty, ratio)


def select_tokens_stratified(
    difficulty: torch.Tensor,
    total_ratio: float,
    num_strata: int = 4,
) -> torch.Tensor:
    """Stratified token selection across difficulty buckets."""
    return _pytorch_stratified_select(difficulty, total_ratio, num_strata)


@triton.jit
def _sequence_coverage_kernel(
    difficulty_ptr,
    mask_ptr,
    batch_size,
    seq_len,
    min_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= batch_size:
        return
        
    row_start = row_idx * seq_len
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    diffs = tl.load(difficulty_ptr + row_start + offsets, mask=mask, other=-1e10)
    masks = tl.load(mask_ptr + row_start + offsets, mask=mask, other=0.0)
    
    selected_count = tl.sum(masks, axis=0).to(tl.int32)
    
    if selected_count < min_tokens:
        num_needed = min_tokens - selected_count
        for _ in range(num_needed):
            eligible = tl.where(masks == 0.0, diffs, -1e10)
            best_idx = tl.argmax(eligible, axis=0)
            masks = tl.where(tl.arange(0, BLOCK_SIZE) == best_idx, 1.0, masks)
            
        tl.store(mask_ptr + row_start + offsets, masks, mask=mask)

@triton.jit
def _stratified_mask_kernel(
    difficulty_ptr,
    mask_ptr,
    thresholds_ptr, # [num_strata]
    strata_counts_ptr, # [num_strata]
    num_tokens,
    num_strata: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_tokens
    
    diffs = tl.load(difficulty_ptr + offsets, mask=mask, other=-1e10)
    
    # For each token, determine if it passes its stratum's threshold
    # We assign tokens to strata based on their values vs thresholds
    # This is an approximation of rank-based stratified
    
    # We'll use a simpler approach for the kernel:
    # Just a parallel thresholding based on pre-computed value thresholds.
    # The complexity is in the wrapper.
    pass

def select_tokens_stratified(
    difficulty: torch.Tensor,
    total_ratio: float,
    num_strata: int = 4,
) -> torch.Tensor:
    """Stratified token selection. Optimized for Triton."""
    if not HAS_TRITON or not difficulty.is_cuda:
        return _pytorch_stratified_select(difficulty, total_ratio, num_strata)
    
    try:
        # Approximate rank-based stratified using histogram thresholds
        flat = difficulty.view(-1)
        num_tokens = flat.numel()
        
        # 1. Get thresholds using quantile (Approximate but fast)
        # We want to divide tokens into num_strata buckets
        q = torch.linspace(0, 1, num_strata + 1, device=difficulty.device)
        quantiles = torch.quantile(flat, q)
        
        # 2. For each stratum, compute how many to select
        # weight = (num_strata - i) / sum(range(1, num_strata + 1))
        # This part is easier in PyTorch as it's O(num_strata)
        total_select = int(num_tokens * total_ratio)
        weights = torch.arange(num_strata, 0, -1, device=difficulty.device, dtype=torch.float32)
        weights /= weights.sum()
        
        n_selects = (total_select * weights).long()
        
        # 3. Compute per-stratum thresholds
        # To be purely rank-based without full sorting, we'd need to sort within buckets.
        # But we can approximate by taking the top-n from each bucket value range.
        
        mask = torch.zeros_like(flat)
        for i in range(num_strata):
            lower = quantiles[num_strata - 1 - i]
            upper = quantiles[num_strata - i]
            
            # Tokens in this value bucket
            bucket_mask = (flat >= lower) & (flat < upper) if i < num_strata - 1 else (flat >= lower)
            if bucket_mask.any():
                bucket_vals = flat[bucket_mask]
                n = min(n_selects[i].item(), bucket_vals.numel())
                if n > 0:
                    stratum_threshold = torch.topk(bucket_vals, n).values[-1]
                    mask[bucket_mask & (flat >= stratum_threshold)] = 1.0
        
        return mask.view(difficulty.shape)
    except Exception:
        return _pytorch_stratified_select(difficulty, total_ratio, num_strata)

def ensure_sequence_coverage(
    difficulty: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
    min_per_seq: int = 1,
) -> torch.Tensor:
    """Ensure minimum token coverage per sequence. Optimized with Triton."""
    if not HAS_TRITON or not difficulty.is_cuda:
        return _pytorch_ensure_sequence_coverage(difficulty, mask, batch_size, seq_len, min_per_seq)
        
    try:
        # Use Triton kernel to avoid batch loop
        # BLOCK_SIZE must be >= seq_len or we need a multi-block approach
        BLOCK_SIZE = triton.next_power_of_2(seq_len)
        if BLOCK_SIZE > 16384: # Limit for single block reduction
            return _pytorch_ensure_sequence_coverage(difficulty, mask, batch_size, seq_len, min_per_seq)
            
        grid = (batch_size,)
        _sequence_coverage_kernel[grid](
            difficulty, mask, batch_size, seq_len, min_per_seq,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return mask
    except Exception:
        return _pytorch_ensure_sequence_coverage(difficulty, mask, batch_size, seq_len, min_per_seq)

def _pytorch_ensure_sequence_coverage(difficulty, mask, batch_size, seq_len, min_per_seq):
    """Fallback implementation."""
    mask_2d = mask.view(batch_size, seq_len)
    diff_2d = difficulty.view(batch_size, seq_len)
    
    for b in range(batch_size):
        selected = mask_2d[b].sum().item()
        if selected < min_per_seq:
            need = int(min_per_seq - selected)
            unselected = (mask_2d[b] == 0)
            if unselected.any():
                scores = diff_2d[b].clone()
                scores[~unselected] = float('-inf')
                _, top_idx = torch.topk(scores, min(need, unselected.sum().item()))
                mask_2d[b, top_idx] = 1.0
    
    return mask_2d.view(mask.shape)

def apply_mask_to_loss(
    per_token_loss: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply selection mask to loss."""
    return per_token_loss * mask.view(per_token_loss.shape)


# Legacy compatibility
def triton_fused_difficulty_score(logits):
    return fused_difficulty_score(logits, targets=None)


class TritonGradientMask(torch.autograd.Function):
    """Legacy - kept for backward compatibility."""
    @staticmethod
    def forward(ctx, tensor, stop_layers, layer_idx, leak_rate):
        return tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None
