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
        
        # Pass 3: Compute entropy and confidence
        entropy_acc = 0.0
        top1_prob = 0.0
        
        for block_start in range(0, vocab_size, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < vocab_size
            vals = tl.load(logits_ptr + row_start + offsets, mask=mask, other=float('-inf'))
            exp_vals = tl.exp(vals - max_val)
            probs = exp_vals / exp_sum
            
            log_probs = tl.log(probs + 1e-10)
            entropy_acc += tl.sum(-probs * log_probs, axis=0)
            block_max = tl.max(probs, axis=0)
            top1_prob = tl.maximum(top1_prob, block_max)
        
        difficulty = entropy_acc - top1_prob
        
        if has_targets:
            target_idx = tl.load(targets_ptr + row_idx)
            target_logit = tl.load(logits_ptr + row_start + target_idx)
            log_sum_exp = tl.log(exp_sum) + max_val
            nll = log_sum_exp - target_logit
            difficulty = difficulty + nll
        
        tl.store(difficulty_ptr + row_idx, difficulty)
        tl.store(confidence_ptr + row_idx, top1_prob)
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
) -> tuple:
    """Compute difficulty scores. Uses Triton if available, else PyTorch."""
    
    if HAS_TRITON and logits.is_cuda:
        try:
            return _triton_difficulty_score(logits, targets)
        except Exception:
            # Fallback on any Triton error
            pass
    
    return _pytorch_difficulty_score(logits, targets)


def _triton_difficulty_score(logits: torch.Tensor, targets: torch.Tensor = None):
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
    """Compute dynamic token ratio based on batch confidence."""
    mean_conf = confidence.mean().item()
    adjusted_ratio = base_ratio * (1.0 + (1.0 - mean_conf) * sensitivity)
    return min(1.0, max(base_ratio * 0.5, adjusted_ratio))


def select_tokens_topk(difficulty: torch.Tensor, ratio: float) -> torch.Tensor:
    """Top-k hardest tokens selection."""
    return _pytorch_select_topk(difficulty, ratio)


def select_tokens_stratified(
    difficulty: torch.Tensor,
    total_ratio: float,
    num_strata: int = 4,
) -> torch.Tensor:
    """Stratified token selection across difficulty buckets."""
    return _pytorch_stratified_select(difficulty, total_ratio, num_strata)


def ensure_sequence_coverage(
    difficulty: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
    min_per_seq: int = 1,
) -> torch.Tensor:
    """Ensure minimum token coverage per sequence."""
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
