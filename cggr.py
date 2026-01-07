"""
CGGR - Confidence-Gated Gradient Routing
=========================================
Selective loss computation with multiple strategies.
All operations accelerated with fused Triton kernels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, List

from triton_kernels import (
    fused_difficulty_score,
    compute_dynamic_threshold,
    select_tokens_topk,
    select_tokens_stratified,
    ensure_sequence_coverage,
    apply_mask_to_loss,
)


class CGGRLoss(nn.Module):
    """
    Advanced selective loss with multiple strategies.
    
    Args:
        scoring: Difficulty scoring method
            - 'entropy': Pure entropy-based
            - 'margin': Margin between top-2 predictions  
            - 'loss': Use per-token loss directly
            - 'combined': Entropy + margin + loss (default)
        
        selection: Token selection strategy
            - 'topk': Top-k hardest tokens
            - 'stratified': Sample from difficulty buckets
            - 'sequence_aware': Ensure coverage per sequence
        
        dynamic_threshold: Adjust ratio based on batch confidence
        threshold_sensitivity: How much to adjust (0-1)
        
        min_tokens_ratio: Target fraction of tokens to keep
        warmup_steps: Steps to reach target sparsity
        
        num_strata: Buckets for stratified sampling
        min_tokens_per_sequence: Minimum coverage per sequence
    """
    
    def __init__(
        self,
        scoring: Literal['entropy', 'margin', 'loss', 'combined'] = 'combined',
        selection: Literal['topk', 'stratified', 'sequence_aware'] = 'topk',
        dynamic_threshold: bool = True,
        threshold_sensitivity: float = 0.5,
        min_tokens_ratio: float = 0.25,
        warmup_steps: int = 1000,
        num_strata: int = 4,
        min_tokens_per_sequence: int = 1,
        base_loss: nn.Module = None,
    ):
        super().__init__()
        
        self.scoring = scoring
        self.selection = selection
        self.dynamic_threshold = dynamic_threshold
        self.threshold_sensitivity = threshold_sensitivity
        self.min_tokens_ratio = min_tokens_ratio
        self.warmup_steps = warmup_steps
        self.num_strata = num_strata
        self.min_tokens_per_sequence = min_tokens_per_sequence
        self.base_loss = base_loss or nn.CrossEntropyLoss(reduction='none')
        
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        self.metrics = {}
    
    def step(self):
        """Call after optimizer.step()"""
        self.step_count += 1
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute selective loss.
        
        Args:
            logits: (batch, seq, vocab) or (N, vocab)
            targets: (batch, seq) or (N,)
        """
        # Handle shapes
        if logits.dim() == 3:
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets.view(-1)
        else:
            batch_size, seq_len = 1, logits.shape[0]
            vocab_size = logits.shape[-1]
            logits_flat = logits
            targets_flat = targets
        
        num_tokens = logits_flat.shape[0]
        
        # Compute per-token loss
        per_token_loss = self.base_loss(logits_flat, targets_flat)
        
        # Compute difficulty scores (Triton-accelerated)
        with torch.no_grad():
            # Choose scoring method
            if self.scoring == 'loss':
                difficulty = per_token_loss.detach()
                confidence = torch.zeros_like(difficulty)
                entropy = difficulty
            else:
                use_targets = self.scoring in ['combined', 'loss']
                difficulty, confidence, entropy = fused_difficulty_score(
                    logits_flat.unsqueeze(0) if logits_flat.dim() == 2 else logits_flat,
                    targets=targets_flat if use_targets else None,
                )
                difficulty = difficulty.view(-1)
                confidence = confidence.view(-1)
                entropy = entropy.view(-1)
            
            # Compute current ratio with curriculum
            if self.warmup_steps <= 0:
                # No warmup - use target ratio immediately
                progress = 1.0
            else:
                progress = min(1.0, self.step_count.item() / self.warmup_steps)
            base_ratio = 1.0 - progress * (1.0 - self.min_tokens_ratio)
            
            # Dynamic threshold adjustment
            if self.dynamic_threshold:
                current_ratio = compute_dynamic_threshold(
                    confidence, base_ratio, self.threshold_sensitivity
                )
            else:
                current_ratio = base_ratio
            
            # Token selection based on strategy
            if self.selection == 'stratified':
                mask = select_tokens_stratified(
                    difficulty, current_ratio, self.num_strata
                )
            elif self.selection == 'sequence_aware':
                mask = select_tokens_topk(difficulty, current_ratio)
                mask = ensure_sequence_coverage(
                    difficulty, mask, batch_size, seq_len, 
                    self.min_tokens_per_sequence
                )
            else:  # topk
                mask = select_tokens_topk(difficulty, current_ratio)
            
            mask = mask.view(-1)
            tokens_selected = mask.sum().item()
        
        # Apply mask to loss (Triton-accelerated)
        masked_loss = apply_mask_to_loss(per_token_loss, mask)
        
        # Average over selected tokens
        loss = masked_loss.sum() / max(tokens_selected, 1)
        
        # Metrics
        self.metrics = {
            'step': self.step_count.item(),
            'token_ratio': current_ratio,
            'tokens_selected': int(tokens_selected),
            'tokens_total': num_tokens,
            'avg_confidence': confidence.mean().item(),
            'avg_entropy': entropy.mean().item(),
            'avg_difficulty': difficulty.mean().item(),
            'selection': self.selection,
            'scoring': self.scoring,
        }
        
        return loss
    
    def get_metrics(self) -> dict:
        return self.metrics.copy()


class CGGRWrapper(nn.Module):
    """
    Legacy wrapper - provides telemetry only.
    Use CGGRLoss for actual compute savings.
    """
    
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.register_buffer('current_step_count', torch.tensor(0, dtype=torch.long))
        self.metrics = {}
        
        import warnings
        warnings.warn(
            "CGGRWrapper is deprecated. Use CGGRLoss for compute savings.",
            DeprecationWarning
        )
    
    def step(self):
        self.current_step_count += 1
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def get_metrics(self):
        return self.metrics
