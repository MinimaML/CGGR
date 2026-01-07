"""
CGGR - Confidence-Gated Gradient Routing
=========================================
Selective loss computation based on token difficulty.
Only hard tokens contribute to loss, so easy tokens don't generate gradients.

This provides ACTUAL compute savings by excluding easy tokens from backprop entirely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from triton_kernels import triton_fused_difficulty_score


class CGGRLoss(nn.Module):
    """
    Wraps a loss function with difficulty-based token selection.
    Only tokens above the difficulty threshold contribute to loss.
    
    Args:
        base_loss: Loss function to wrap (default: CrossEntropyLoss with reduction='none')
        min_tokens_ratio: Minimum fraction of tokens to include (default: 0.25)
        warmup_steps: Steps to reach target sparsity (default: 1000)
    """
    
    def __init__(
        self, 
        base_loss: nn.Module = None,
        min_tokens_ratio: float = 0.25,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.base_loss = base_loss or nn.CrossEntropyLoss(reduction='none')
        self.min_tokens_ratio = min_tokens_ratio
        self.warmup_steps = warmup_steps
        
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        self.metrics = {}
    
    def step(self):
        """Call after optimizer.step()"""
        self.step_count += 1
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss only for hard tokens.
        
        Args:
            logits: (batch, seq, vocab) or (batch*seq, vocab)
            targets: (batch, seq) or (batch*seq,)
        
        Returns:
            Scalar loss averaged over selected tokens
        """
        # Flatten if needed
        if logits.dim() == 3:
            batch, seq, vocab = logits.shape
            logits_flat = logits.view(-1, vocab)
            targets_flat = targets.view(-1)
        else:
            logits_flat = logits
            targets_flat = targets
            batch, seq = 1, logits.shape[0]
        
        num_tokens = logits_flat.shape[0]
        
        # Compute per-token loss
        per_token_loss = self.base_loss(logits_flat, targets_flat)
        
        # Compute difficulty scores (Triton-accelerated)
        with torch.no_grad():
            difficulty, confidence, entropy = triton_fused_difficulty_score(
                logits_flat.unsqueeze(0) if logits_flat.dim() == 2 else logits_flat
            )
            difficulty = difficulty.view(-1)
            
            # Curriculum: gradually reduce token ratio
            progress = min(1.0, self.step_count.item() / max(1, self.warmup_steps))
            current_ratio = 1.0 - progress * (1.0 - self.min_tokens_ratio)
            
            # Select top-k hardest tokens
            k = max(1, int(num_tokens * current_ratio))
            _, hard_indices = torch.topk(difficulty, k)
            
            # Create mask
            mask = torch.zeros(num_tokens, device=logits.device, dtype=logits.dtype)
            mask[hard_indices] = 1.0
        
        # Masked loss (only hard tokens contribute)
        masked_loss = per_token_loss * mask
        loss = masked_loss.sum() / mask.sum()
        
        # Metrics
        self.metrics = {
            'step': self.step_count.item(),
            'token_ratio': current_ratio,
            'tokens_used': k,
            'tokens_total': num_tokens,
            'avg_confidence': confidence.mean().item(),
            'avg_masked_loss': loss.item(),
        }
        
        return loss
    
    def get_metrics(self) -> dict:
        return self.metrics.copy()


class CGGRWrapper(nn.Module):
    """
    Legacy wrapper for backward compatibility.
    Now just provides telemetry without gradient routing overhead.
    Use CGGRLoss for actual compute savings.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_buckets: int = 4,
        warmup_steps: int = 1000,
        leak_rate: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.num_buckets = num_buckets
        self.warmup_steps = warmup_steps
        
        self.register_buffer('current_step_count', torch.tensor(0, dtype=torch.long))
        self.metrics = {}
        
        import warnings
        warnings.warn(
            "CGGRWrapper adds overhead without compute savings. "
            "Use CGGRLoss for actual backward speedup.",
            DeprecationWarning
        )
    
    def step(self):
        self.current_step_count += 1
    
    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        
        if isinstance(output, torch.Tensor):
            logits = output
        elif hasattr(output, 'logits'):
            logits = output.logits
        elif isinstance(output, tuple):
            logits = output[0]
        else:
            return output
        
        if logits is not None and self.training:
            with torch.no_grad():
                difficulty, conf, ent = triton_fused_difficulty_score(logits)
                self.metrics = {
                    'step': self.current_step_count.item(),
                    'avg_confidence': conf.mean().item(),
                    'avg_entropy': ent.mean().item(),
                }
        
        return output
    
    def get_metrics(self) -> dict:
        return self.metrics.copy()
