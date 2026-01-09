"""
CGGR Selective Gradient Checkpointing
======================================
Only apply activation checkpointing to sequences/tokens
that will participate in the backward pass (hard tokens).
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, List, Callable
from contextlib import contextmanager


class SelectiveCheckpointWrapper(nn.Module):
    """
    Wraps a model to apply gradient checkpointing only to selected sequences.
    
    Usage:
        wrapper = SelectiveCheckpointWrapper(model)
        loss = wrapper(input_ids, labels, hard_indices=[0, 2, 5])
    """
    
    def __init__(self, model: nn.Module, checkpoint_layers: Optional[List[int]] = None):
        super().__init__()
        self.model = model
        self.checkpoint_layers = checkpoint_layers  # If None, checkpoint all
        self._checkpointing_enabled = False
        
    def enable_checkpointing(self):
        """Enable gradient checkpointing on the underlying model."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            self._checkpointing_enabled = True
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'gradient_checkpointing_enable'):
            self.model.model.gradient_checkpointing_enable()
            self._checkpointing_enabled = True
            
    def disable_checkpointing(self):
        """Disable gradient checkpointing on the underlying model."""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
            self._checkpointing_enabled = False
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'gradient_checkpointing_disable'):
            self.model.model.gradient_checkpointing_disable()
            self._checkpointing_enabled = False
    
    @contextmanager
    def selective_checkpointing(self, enable: bool = True):
        """Context manager for temporary checkpointing state."""
        was_enabled = self._checkpointing_enabled
        try:
            if enable:
                self.enable_checkpointing()
            else:
                self.disable_checkpointing()
            yield
        finally:
            if was_enabled:
                self.enable_checkpointing()
            else:
                self.disable_checkpointing()
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        hard_indices: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass with selective checkpointing.
        
        Args:
            input_ids: (batch, seq) input tokens
            labels: (batch, seq) target tokens
            hard_indices: Indices of sequences to apply checkpointing to.
                         If None, processes all sequences without checkpointing.
        """
        if hard_indices is None or len(hard_indices) == 0:
            # No hard sequences specified, run without checkpointing
            return self.model(input_ids, labels=labels, **kwargs)
        
        batch_size = input_ids.shape[0]
        
        # Split into hard and easy sequences
        all_indices = torch.arange(batch_size, device=input_ids.device)
        easy_mask = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
        easy_mask[hard_indices] = False
        easy_indices = all_indices[easy_mask]
        
        outputs = {}
        total_loss = 0.0
        
        # Process EASY sequences WITHOUT checkpointing (no backward needed anyway)
        if len(easy_indices) > 0:
            with self.selective_checkpointing(enable=False):
                with torch.no_grad():  # No grad for easy sequences
                    easy_input = input_ids[easy_indices]
                    _ = self.model(easy_input, **kwargs)
        
        # Process HARD sequences WITH checkpointing
        if len(hard_indices) > 0:
            with self.selective_checkpointing(enable=True):
                hard_input = input_ids[hard_indices]
                hard_labels = labels[hard_indices] if labels is not None else None
                hard_outputs = self.model(hard_input, labels=hard_labels, **kwargs)
                
                if hasattr(hard_outputs, 'loss') and hard_outputs.loss is not None:
                    total_loss = hard_outputs.loss
                else:
                    # Manual loss computation
                    logits = hard_outputs.logits if hasattr(hard_outputs, 'logits') else hard_outputs
                    if hard_labels is not None:
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = hard_labels[..., 1:].contiguous()
                        total_loss = nn.functional.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                outputs = hard_outputs
        
        # Return loss or outputs
        if hasattr(outputs, 'loss'):
            outputs.loss = total_loss
            return outputs
        return total_loss


def apply_selective_checkpointing(
    model: nn.Module,
    forward_fn: Callable,
    input_ids: torch.Tensor,
    hard_indices: torch.Tensor,
    **kwargs
):
    """
    Functional API for selective checkpointing.
    
    Args:
        model: The model to run
        forward_fn: Custom forward function (e.g., model.forward)
        input_ids: Full batch of inputs
        hard_indices: Indices of hard sequences
        **kwargs: Additional forward args
    
    Returns:
        Outputs from hard sequences only (with gradients)
    """
    hard_input = input_ids[hard_indices]
    
    # Enable checkpointing for this call
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    def run_forward(*args):
        return forward_fn(*args, **kwargs)
    
    # Use torch checkpoint for memory efficiency
    if hard_input.requires_grad or any(p.requires_grad for p in model.parameters()):
        outputs = checkpoint(run_forward, hard_input, use_reentrant=False)
    else:
        outputs = run_forward(hard_input)
    
    return outputs


class CGGRCheckpointedModel(nn.Module):
    """
    Drop-in replacement for CGGRModel with integrated selective checkpointing.
    
    Combines CGGR's batch splitting with selective gradient checkpointing,
    only recomputing activations for sequences that will have gradients.
    """
    
    def __init__(
        self,
        model: nn.Module,
        router: Optional[nn.Module] = None,
        min_tokens_ratio: float = 0.25,
        warmup_steps: int = 1000,
        use_checkpointing: bool = True,
    ):
        super().__init__()
        # Import here to avoid circular dependency
        from cggr import CGGRScorer, create_truncated_router
        
        self.model = model
        self.use_checkpointing = use_checkpointing
        
        router_model = router if router is not None else model
        self.scorer = CGGRScorer(
            router=router_model,
            min_tokens_ratio=min_tokens_ratio,
            warmup_steps=warmup_steps,
            selection='fixed_quota',  # Deterministic for checkpointing
        )
        self.metrics = {}
        
    def step(self):
        self.scorer.step()
    
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        if labels is None:
            return self.model(input_ids, **kwargs)
        
        batch_size, seq_len = input_ids.shape
        
        # PASS 1: Score difficulty
        difficulty, mask, info = self.scorer(input_ids, **kwargs)
        current_ratio = info['current_ratio']
        
        # Select hard sequences
        seq_difficulty = difficulty.view(batch_size, seq_len).mean(dim=-1)
        k = max(1, int(batch_size * current_ratio))
        _, hard_seq_indices = torch.topk(seq_difficulty, k)
        
        # PASS 2: Forward with selective checkpointing
        hard_input_ids = input_ids[hard_seq_indices]
        hard_labels = labels[hard_seq_indices]
        
        if self.use_checkpointing:
            # Enable checkpointing only for this forward pass
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
        
        hard_outputs = self.model(hard_input_ids, **kwargs)
        
        if self.use_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_disable'):
                self.model.gradient_checkpointing_disable()
        
        # Compute loss
        if hasattr(hard_outputs, 'logits'):
            hard_logits = hard_outputs.logits
        else:
            hard_logits = hard_outputs
        
        shift_logits = hard_logits[:, :-1, :].contiguous().view(-1, hard_logits.shape[-1])
        shift_labels = hard_labels[:, 1:].contiguous().view(-1)
        loss = nn.functional.cross_entropy(shift_logits, shift_labels)
        
        # Metrics
        self.metrics = {
            'step': self.scorer.step_count.item(),
            'token_ratio': current_ratio,
            'tokens_selected': k * (seq_len - 1),
            'tokens_total': batch_size * seq_len,
            'checkpointing_enabled': self.use_checkpointing,
        }
        
        return loss
    
    def get_metrics(self) -> dict:
        return self.metrics.copy()
