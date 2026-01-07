"""
CGGR - Confidence-Gated Gradient Routing
=========================================
A PyTorch library for selective gradient routing based on token difficulty.
Requires Triton for CUDA kernel acceleration.
"""

import torch
import torch.nn as nn

from triton_kernels import (
    triton_fused_difficulty_score,
    triton_bucket_assign,
    TritonGradientMask,
)


class GradientRouter:
    """Routes gradients based on per-token stop layers using Triton."""
    
    def __init__(self, layer_idx: int, leak_rate: float = 0.0):
        self.layer_idx = layer_idx
        self.leak_rate = leak_rate
        self.stop_layer_tensor = None

    def set_stop_layers(self, stop_layer_tensor: torch.Tensor):
        """Set stop layers for each token. Shape: (batch, seq) or (batch, seq, 1)"""
        self.stop_layer_tensor = stop_layer_tensor

    def apply_to_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply gradient masking via custom autograd function."""
        if self.stop_layer_tensor is None:
            return tensor
        return TritonGradientMask.apply(
            tensor, self.stop_layer_tensor, self.layer_idx, self.leak_rate
        )


class CGGRWrapper(nn.Module):
    """
    Wraps a Transformer model with Confidence-Gated Gradient Routing.
    
    Requires Triton for optimal performance on CUDA devices.
    
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
        
        # Persistence: step count saved with model
        self.register_buffer('current_step_count', torch.tensor(0, dtype=torch.long))
        
        # Telemetry
        self.metrics = {}
        
        # Detect transformer layers
        self.layer_module_list = self._detect_transformer_layers(model)
        if not self.layer_module_list:
            raise ValueError("Could not detect Transformer layer stack (ModuleList)")
        self.num_layers = len(self.layer_module_list)
        
        # Pre-compute bucket -> stop_layer mapping
        self._bucket_stop_layers = self._compute_bucket_stop_layers()
        
        # Setup routers
        self.routers = {}  # layer_idx -> GradientRouter
        self._setup_routers()
    
    def _compute_bucket_stop_layers(self) -> list:
        """Compute stop layer for each bucket index."""
        chunk_size = self.num_layers / self.num_buckets
        stop_layers = []
        for b_idx in range(self.num_buckets):
            layers_to_keep = int(chunk_size * (self.num_buckets - b_idx))
            stop_layer = self.num_layers - layers_to_keep - 1
            stop_layers.append(stop_layer)
        return stop_layers
    
    def _setup_routers(self):
        """Create routers at gradient cutoff points."""
        for b_idx in range(self.num_buckets):
            stop_layer = self._bucket_stop_layers[b_idx]
            if stop_layer >= 0 and stop_layer not in self.routers:
                self.routers[stop_layer] = GradientRouter(stop_layer, leak_rate=self.leak_rate)
    
    def _detect_transformer_layers(self, model: nn.Module):
        """Find the largest ModuleList (assumed to be transformer layers)."""
        largest = None
        max_len = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > max_len:
                max_len = len(module)
                largest = module
        return largest
    
    def step(self):
        """Increment training step counter."""
        self.current_step_count += 1
    
    def forward(self, *args, **kwargs):
        """Forward pass with Triton-accelerated gradient routing."""
        # Run original forward
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
        
        # Compute difficulty and assign buckets (Triton)
        with torch.no_grad():
            progress = min(1.0, self.current_step_count.item() / max(1, self.warmup_steps))
            difficulty, conf, ent = triton_fused_difficulty_score(logits)
            
            token_stop_layers = triton_bucket_assign(
                difficulty, self.num_buckets, self.num_layers, progress
            )
            
            # Update routers
            for router in self.routers.values():
                router.set_stop_layers(token_stop_layers)
            
            # Metrics
            self.metrics = {
                'step': self.current_step_count.item(),
                'avg_stop_layer': token_stop_layers.float().mean().item(),
                'avg_confidence': conf.mean().item(),
                'avg_entropy': ent.mean().item(),
            }
        
        return output
    
    def set_manual_stop_layers(self, stop_layers: torch.Tensor):
        """Manually set stop layers for all tokens (for testing)."""
        for router in self.routers.values():
            router.set_stop_layers(stop_layers)
    
    def get_metrics(self) -> dict:
        """Get current routing metrics."""
        return self.metrics.copy()
