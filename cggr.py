"""
CGGR - Confidence-Gated Gradient Routing
=========================================
A PyTorch library for selective gradient routing based on token difficulty.
Now with optional Triton kernel acceleration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Triton kernels
try:
    from triton_kernels import (
        HAS_TRITON,
        triton_fused_difficulty_score,
        triton_bucket_assign,
        apply_triton_gradient_mask,
    )
except ImportError:
    HAS_TRITON = False
    triton_fused_difficulty_score = None
    triton_bucket_assign = None
    apply_triton_gradient_mask = None


def compute_difficulty_score(logits: torch.Tensor):
    """
    Compute difficulty, confidence, and entropy from logits.
    Uses Triton kernel if available on CUDA, otherwise PyTorch.
    """
    if HAS_TRITON and logits.is_cuda:
        return triton_fused_difficulty_score(logits)
    
    # PyTorch fallback
    probs = F.softmax(logits, dim=-1)
    confidence, _ = torch.max(probs, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    difficulty = entropy - confidence
    return difficulty, confidence, entropy


class GradientRouter:
    """Routes gradients based on per-token stop layers."""
    
    def __init__(self, layer_idx: int, leak_rate: float = 0.0):
        self.layer_idx = layer_idx
        self.leak_rate = leak_rate
        self.stop_layer_tensor = None

    def set_stop_layers(self, stop_layer_tensor: torch.Tensor):
        """Set stop layers for each token. Shape: (batch, seq) or (batch, seq, 1)"""
        self.stop_layer_tensor = stop_layer_tensor

    def gradient_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward hook that masks gradients based on stop layers."""
        if self.stop_layer_tensor is None:
            return grad
        
        stop = self.stop_layer_tensor
        if stop.device != grad.device:
            stop = stop.to(grad.device)
        if stop.dim() == 2:
            stop = stop.unsqueeze(-1)
            
        should_pass = (stop < self.layer_idx).float()
        
        if self.leak_rate > 0:
            mask = should_pass * (1.0 - self.leak_rate) + self.leak_rate
        else:
            mask = should_pass
        
        return grad * mask

    def register_hook(self, tensor: torch.Tensor):
        """Register backward hook on tensor."""
        if tensor.requires_grad:
            tensor.register_hook(self.gradient_hook)


class CGGRWrapper(nn.Module):
    """
    Wraps a Transformer model with Confidence-Gated Gradient Routing.
    
    Args:
        model: The model to wrap (must have a ModuleList of transformer layers)
        num_buckets: Number of gradient depth tiers (default: 4)
        warmup_steps: Steps to gradually enable routing (default: 1000)
        leak_rate: Fraction of gradient to leak through blocked paths (default: 0.0)
        use_triton: Force Triton on/off. None = auto-detect (default: None)
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_buckets: int = 4,
        warmup_steps: int = 1000,
        leak_rate: float = 0.0,
        use_triton: bool = None,
    ):
        super().__init__()
        self.model = model
        self.num_buckets = num_buckets
        self.warmup_steps = warmup_steps
        self.leak_rate = leak_rate
        
        # Triton mode
        self._use_triton = use_triton if use_triton is not None else HAS_TRITON
        
        # Persistence: step count saved with model
        self.register_buffer('current_step_count', torch.tensor(0, dtype=torch.long))
        
        # Telemetry
        self.metrics = {}
        
        # Detect transformer layers
        self.layer_module_list = self._detect_transformer_layers(model)
        if not self.layer_module_list:
            raise ValueError("Could not detect Transformer layer stack (ModuleList)")
        self.num_layers = len(self.layer_module_list)
        
        # Pre-compute bucket -> stop_layer mapping (cached)
        self._bucket_stop_layers = self._compute_bucket_stop_layers()
        self.register_buffer(
            '_bucket_stop_layers_tensor',
            torch.tensor(self._bucket_stop_layers, dtype=torch.long)
        )
        
        # Setup routers and hooks
        self.routers = []
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
        """Create routers and register hooks at gradient cutoff points."""
        created_layers = set()
        
        for b_idx in range(self.num_buckets):
            stop_layer = self._bucket_stop_layers[b_idx]
            
            if stop_layer >= 0 and stop_layer not in created_layers:
                router = GradientRouter(stop_layer, leak_rate=self.leak_rate)
                self.routers.append(router)
                created_layers.add(stop_layer)
                
                target_module = self.layer_module_list[stop_layer]
                self._register_forward_hook(target_module, router)
    
    def _register_forward_hook(self, module: nn.Module, router: GradientRouter):
        """Register forward hook to attach gradient hook on layer output."""
        def hook(mod, inp, out):
            if not self.training:
                return
            tensor = out[0] if isinstance(out, tuple) else out
            if isinstance(tensor, torch.Tensor):
                router.register_hook(tensor)
        module.register_forward_hook(hook)
    
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
        """Forward pass with difficulty-based gradient routing."""
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
        
        # Compute difficulty and route gradients
        with torch.no_grad():
            # Curriculum progress
            progress = min(1.0, self.current_step_count.item() / max(1, self.warmup_steps))
            
            # Compute difficulty scores
            difficulty, conf, ent = compute_difficulty_score(logits)
            batch_size, seq_len = difficulty.shape
            
            # Assign buckets (Triton or PyTorch)
            if self._use_triton and HAS_TRITON and difficulty.is_cuda:
                token_stop_layers = triton_bucket_assign(
                    difficulty, self.num_buckets, self.num_layers, progress
                ).unsqueeze(-1)
            else:
                # PyTorch fallback: rank-based bucketing
                flat_diff = difficulty.view(-1)
                ranks = torch.argsort(torch.argsort(flat_diff))
                num_tokens = flat_diff.numel()
                
                bucket_indices = (self.num_buckets - 1) - (ranks * self.num_buckets // num_tokens)
                bucket_indices = bucket_indices.clamp(0, self.num_buckets - 1)
                
                # Apply curriculum
                if self.warmup_steps > 0:
                    allowed_max = int(progress * (self.num_buckets - 1))
                    bucket_indices = torch.min(
                        bucket_indices,
                        torch.tensor(allowed_max, device=difficulty.device)
                    )
                
                # Map bucket -> stop layer
                lookup = self._bucket_stop_layers_tensor.to(difficulty.device)
                token_stop_layers = lookup[bucket_indices].view(batch_size, seq_len, 1)
            
            # Update routers
            for router in self.routers:
                router.set_stop_layers(token_stop_layers)
            
            # Metrics
            self.metrics = {
                'step': self.current_step_count.item(),
                'avg_bucket': token_stop_layers.float().mean().item(),
                'avg_confidence': conf.mean().item(),
                'avg_entropy': ent.mean().item(),
                'triton_enabled': self._use_triton and HAS_TRITON,
            }
        
        return output
    
    def set_manual_stop_layers(self, stop_layers: torch.Tensor):
        """Manually set stop layers for all tokens (for testing)."""
        for router in self.routers:
            router.set_stop_layers(stop_layers)
    
    def get_metrics(self) -> dict:
        """Get current routing metrics."""
        return self.metrics.copy()
