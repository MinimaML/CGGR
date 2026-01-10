"""
CGGR Flash Attention Integration
=================================
Utilities for enabling Flash Attention with CGGR models.

Supports:
- PyTorch 2.0+ native SDPA (scaled_dot_product_attention)
- HuggingFace Transformers Flash Attention 2
- flash-attn library (when available)
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Literal
import warnings

# Check available Flash Attention backends
HAS_FLASH_ATTN = False
HAS_SDPA = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

try:
    import flash_attn
    HAS_FLASH_ATTN = True
    FLASH_ATTN_VERSION = getattr(flash_attn, '__version__', 'unknown')
except ImportError:
    FLASH_ATTN_VERSION = None


def get_flash_attention_info() -> dict:
    """
    Get information about available Flash Attention backends.
    
    Returns:
        dict with backend availability and recommendations
    """
    info = {
        'sdpa_available': HAS_SDPA,
        'flash_attn_available': HAS_FLASH_ATTN,
        'flash_attn_version': FLASH_ATTN_VERSION,
        'cuda_available': torch.cuda.is_available(),
        'recommended_backend': None,
    }
    
    if HAS_FLASH_ATTN and torch.cuda.is_available():
        info['recommended_backend'] = 'flash_attention_2'
    elif HAS_SDPA:
        info['recommended_backend'] = 'sdpa'
    else:
        info['recommended_backend'] = 'eager'
    
    return info


def enable_flash_attention(
    model: nn.Module,
    backend: Literal['auto', 'flash_attention_2', 'sdpa', 'eager'] = 'auto',
    verbose: bool = True,
) -> nn.Module:
    """
    Enable Flash Attention on a HuggingFace model.
    
    This function attempts to configure the model for Flash Attention using
    the best available backend.
    
    Args:
        model: HuggingFace model (e.g., LlamaForCausalLM)
        backend: Attention backend to use
            - 'auto': Automatically select best available
            - 'flash_attention_2': Use flash-attn library (requires install)
            - 'sdpa': Use PyTorch's native SDPA
            - 'eager': Standard attention (no Flash Attention)
        verbose: Print backend selection info
    
    Returns:
        The model with Flash Attention enabled
    
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> from cggr_flash import enable_flash_attention
        >>> model = AutoModelForCausalLM.from_pretrained("...")
        >>> model = enable_flash_attention(model)
    """
    # Auto-select backend
    if backend == 'auto':
        info = get_flash_attention_info()
        backend = info['recommended_backend']
        if verbose:
            print(f"[CGGR Flash] Auto-selected backend: {backend}")
    
    # Check if model supports attn_implementation
    if hasattr(model, 'config'):
        config = model.config
        
        # Set attention implementation in config
        if hasattr(config, '_attn_implementation'):
            if backend == 'flash_attention_2' and not HAS_FLASH_ATTN:
                warnings.warn(
                    "flash-attn not installed. Install with: pip install flash-attn --no-build-isolation. "
                    "Falling back to SDPA if available."
                )
                backend = 'sdpa' if HAS_SDPA else 'eager'
            
            config._attn_implementation = backend
            if verbose:
                print(f"[CGGR Flash] Set attention implementation to: {backend}")
        
        # Enable use_cache compatibility with FA2
        if backend == 'flash_attention_2' and hasattr(config, 'use_cache'):
            # FA2 works with use_cache in newer versions
            pass
    
    # Try to use PyTorch's built-in SDPA context manager
    if backend == 'sdpa' and HAS_SDPA:
        # Enable SDPA globally for this model
        _enable_sdpa_on_model(model)
        if verbose:
            print("[CGGR Flash] Enabled PyTorch SDPA")
    
    return model


def _enable_sdpa_on_model(model: nn.Module):
    """Enable SDPA on model's attention layers."""
    # For HuggingFace models, SDPA is typically enabled via config
    # This function handles edge cases
    for name, module in model.named_modules():
        # Check for attention modules that might need SDPA hint
        if 'attention' in name.lower() or 'attn' in name.lower():
            if hasattr(module, 'is_causal'):
                module.is_causal = True


def load_model_with_flash_attention(
    model_name_or_path: str,
    backend: Literal['auto', 'flash_attention_2', 'sdpa', 'eager'] = 'auto',
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, dict]] = 'auto',
    **kwargs,
) -> nn.Module:
    """
    Load a HuggingFace model with Flash Attention enabled.
    
    This is a convenience function that combines model loading with
    Flash Attention configuration.
    
    Args:
        model_name_or_path: Model identifier or path
        backend: Attention backend to use
        torch_dtype: Model dtype (default: auto-detect)
        device_map: Device placement strategy
        **kwargs: Additional arguments for from_pretrained
    
    Returns:
        Model with Flash Attention enabled
    
    Example:
        >>> from cggr_flash import load_model_with_flash_attention
        >>> model = load_model_with_flash_attention(
        ...     "microsoft/phi-2",
        ...     backend="flash_attention_2",
        ...     torch_dtype=torch.float16,
        ... )
    """
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError:
        raise ImportError("transformers library required. Install with: pip install transformers")
    
    # Auto-select backend
    if backend == 'auto':
        info = get_flash_attention_info()
        backend = info['recommended_backend']
    
    # Check flash-attn availability
    if backend == 'flash_attention_2' and not HAS_FLASH_ATTN:
        warnings.warn(
            "flash-attn not installed. Falling back to SDPA. "
            "Install with: pip install flash-attn --no-build-isolation"
        )
        backend = 'sdpa' if HAS_SDPA else 'eager'
    
    # Auto-detect dtype for Flash Attention (requires float16/bfloat16)
    if torch_dtype is None and backend in ['flash_attention_2', 'sdpa']:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    
    # Load model with attention implementation
    print(f"[CGGR Flash] Loading {model_name_or_path} with {backend} attention")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        attn_implementation=backend,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **kwargs,
    )
    
    return model


class FlashAttentionWrapper(nn.Module):
    """
    Wrapper that adds Flash Attention capability to any model.
    
    For models that don't natively support Flash Attention config,
    this wrapper provides a context manager approach.
    
    Usage:
        wrapper = FlashAttentionWrapper(model)
        with wrapper.flash_attention_context():
            output = wrapper(input_ids)
    """
    
    def __init__(self, model: nn.Module, backend: str = 'auto'):
        super().__init__()
        self.model = model
        self.backend = backend
        
        # Apply Flash Attention if possible
        enable_flash_attention(model, backend=backend, verbose=False)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    @staticmethod
    def flash_attention_context():
        """
        Context manager for enabling SDPA during forward pass.
        
        Usage:
            with FlashAttentionWrapper.flash_attention_context():
                output = model(input_ids)
        """
        if HAS_SDPA:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=True,
            )
        else:
            # No-op context manager
            from contextlib import nullcontext
            return nullcontext()


def benchmark_attention_backends(
    model: nn.Module,
    input_ids: torch.Tensor,
    warmup_steps: int = 3,
    benchmark_steps: int = 10,
) -> dict:
    """
    Benchmark different attention backends on a model.
    
    Args:
        model: Model to benchmark
        input_ids: Sample input for benchmarking
        warmup_steps: Number of warmup iterations
        benchmark_steps: Number of timed iterations
    
    Returns:
        dict with timing results for each backend
    """
    import time
    
    results = {}
    
    backends_to_test = ['eager']
    if HAS_SDPA:
        backends_to_test.append('sdpa')
    if HAS_FLASH_ATTN:
        backends_to_test.append('flash_attention_2')
    
    for backend in backends_to_test:
        try:
            # Configure backend
            if hasattr(model, 'config') and hasattr(model.config, '_attn_implementation'):
                model.config._attn_implementation = backend
            
            # Warmup
            for _ in range(warmup_steps):
                with torch.no_grad():
                    _ = model(input_ids)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(benchmark_steps):
                with torch.no_grad():
                    _ = model(input_ids)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            end = time.perf_counter()
            
            avg_time = (end - start) / benchmark_steps * 1000  # ms
            results[backend] = {
                'avg_time_ms': avg_time,
                'throughput': 1000 / avg_time,  # iterations per second
            }
            
        except Exception as e:
            results[backend] = {'error': str(e)}
    
    return results


# Export key components
__all__ = [
    'get_flash_attention_info',
    'enable_flash_attention',
    'load_model_with_flash_attention',
    'FlashAttentionWrapper',
    'benchmark_attention_backends',
    'HAS_FLASH_ATTN',
    'HAS_SDPA',
]
