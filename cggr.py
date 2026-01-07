"""
CGGR - Confidence-Gated Gradient Routing
=========================================
A PyTorch library for selective gradient routing based on token difficulty.
Uses Triton kernels for CUDA acceleration.

Strategy: Uses logits from the PREVIOUS step to route gradients for the CURRENT step.
This avoids needing a second forward pass while still providing adaptive routing.
"""

import torch
import torch.nn as nn

from triton_kernels import (
    triton_fused_difficulty_score,
    triton_bucket_assign,
    TritonGradientMask,
)


class CGGRWrapper(nn.Module):
    """
    Wraps a Transformer model with Confidence-Gated Gradient Routing.
    
    Uses the previous step's difficulty scores to route gradients for the current step.
    This is a one-step-delayed routing that avoids extra forward passes.
    
    Args:
        model: The model to wrap (must have a ModuleList of transformer layers)
        num_buckets: Number of gradient depth tiers (default: 4)
        warmup_steps: Steps to gradually enable routing (default: 1000)
        leak_rate: Fraction of gradient to leak through blocked paths (default: 0.0)
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
        self.leak_rate = leak_rate
        
        # Persistence
        self.register_buffer('current_step_count', torch.tensor(0, dtype=torch.long))
        
        # Detect transformer layers
        self.layer_module_list = self._detect_transformer_layers(model)
        if not self.layer_module_list:
            raise ValueError("Could not detect Transformer layer stack")
        self.num_layers = len(self.layer_module_list)
        
        # Pre-compute bucket -> stop_layer
        self._bucket_stop_layers = self._compute_bucket_stop_layers()
        
        # Cutoff layers
        self._cutoff_layers = sorted(set(
            sl for sl in self._bucket_stop_layers if sl >= 0
        ))
        
        # Routing state: stop layers for NEXT forward pass (from previous step)
        self._pending_stop_layers = None
        self._active_stop_layers = None  # Currently being used
        
        # Register forward hooks
        for layer_idx in self._cutoff_layers:
            self._register_gradient_gate(layer_idx)
        
        self.metrics = {}
    
    def _compute_bucket_stop_layers(self) -> list:
        chunk_size = self.num_layers / self.num_buckets
        stop_layers = []
        for b_idx in range(self.num_buckets):
            layers_to_keep = int(chunk_size * (self.num_buckets - b_idx))
            stop_layer = self.num_layers - layers_to_keep - 1
            stop_layers.append(stop_layer)
        return stop_layers
    
    def _register_gradient_gate(self, layer_idx: int):
        target_layer = self.layer_module_list[layer_idx]
        leak_rate = self.leak_rate
        
        def forward_hook(module, inp, out):
            if not self.training or self._active_stop_layers is None:
                return out
            
            tensor = out[0] if isinstance(out, tuple) else out
            if not isinstance(tensor, torch.Tensor):
                return out
            
            masked = TritonGradientMask.apply(
                tensor, self._active_stop_layers, layer_idx, leak_rate
            )
            
            if isinstance(out, tuple):
                return (masked,) + out[1:]
            return masked
        
        target_layer.register_forward_hook(forward_hook)
    
    def _detect_transformer_layers(self, model):
        largest = None
        max_len = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > max_len:
                max_len = len(module)
                largest = module
        return largest
    
    def step(self):
        """Call after optimizer.step() to advance routing state."""
        self.current_step_count += 1
        # Promote pending routing to active for next forward
        self._active_stop_layers = self._pending_stop_layers
        self._pending_stop_layers = None
    
    def forward(self, *args, **kwargs):
        # Run forward with current routing
        output = self.model(*args, **kwargs)
        
        # Extract logits
        if isinstance(output, torch.Tensor):
            logits = output
        elif hasattr(output, 'logits'):
            logits = output.logits
        elif isinstance(output, tuple):
            logits = output[0]
        else:
            return output
        
        if logits is None or not self.training:
            return output
        
        # Compute routing for NEXT step based on this step's difficulty
        with torch.no_grad():
            progress = min(1.0, self.current_step_count.item() / max(1, self.warmup_steps))
            difficulty, conf, ent = triton_fused_difficulty_score(logits)
            
            self._pending_stop_layers = triton_bucket_assign(
                difficulty, self.num_buckets, self.num_layers, progress
            )
            
            self.metrics = {
                'step': self.current_step_count.item(),
                'routing_active': self._active_stop_layers is not None,
                'avg_confidence': conf.mean().item(),
                'avg_entropy': ent.mean().item(),
            }
            
            if self._active_stop_layers is not None:
                self.metrics['avg_stop_layer'] = self._active_stop_layers.float().mean().item()
        
        return output
    
    def get_metrics(self) -> dict:
        return self.metrics.copy()
