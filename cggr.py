import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientRouter:
    def __init__(self, layer_idx, leak_rate=0.0):
        self.layer_idx = layer_idx
        self.leak_rate = leak_rate
        self.stop_layer_tensor = None
        self.hook_handle = None

    def set_stop_layers(self, stop_layer_tensor):
        """
        Stores the tensor of stop layers for each token.
        stop_layer_tensor: (batch_size, seq_len, 1) or (batch_size, seq_len)
        Values are the integer layer index where gradients should STOP.
        """
        self.stop_layer_tensor = stop_layer_tensor

    def gradient_hook(self, grad):
        """
        mask = 1 if token_stop_layer < self.layer_idx (Allow full gradient)
        mask = leak_rate if token_stop_layer >= self.layer_idx (Stop/Suppress gradient)
        """
        if self.stop_layer_tensor is None:
            return grad
        
        if self.stop_layer_tensor.device != grad.device:
            self.stop_layer_tensor = self.stop_layer_tensor.to(grad.device)
            
        cutoff_tensor = self.stop_layer_tensor
        if cutoff_tensor.dim() == 2:
            cutoff_tensor = cutoff_tensor.unsqueeze(-1)
            
        # Logic: Pass if stop_layer < current_layer
        # "Passing" means mask = 1.0. 
        # "Blocking" means mask = leak_rate (usually 0.0)
        
        should_pass = (cutoff_tensor < self.layer_idx).float()
        
        # If leak_rate > 0, we add it to the 'blocked' portion
        # Mask = should_pass * 1.0 + (1 - should_pass) * leak_rate
        #      = should_pass + leak_rate - should_pass * leak_rate
        #      = should_pass * (1 - leak_rate) + leak_rate
        
        if self.leak_rate > 0:
             mask = should_pass * (1.0 - self.leak_rate) + self.leak_rate
        else:
             mask = should_pass
        
        return grad * mask

    def register_hook(self, tensor):
        if tensor.requires_grad:
            self.hook_handle = tensor.register_hook(self.gradient_hook)


def compute_difficulty_score(logits):
    probs = F.softmax(logits, dim=-1)
    confidence, _ = torch.max(probs, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    # Higher = Harder
    difficulty = entropy - confidence
    return difficulty, confidence, entropy

class CGGRWrapper(nn.Module):
    def __init__(self, model, num_buckets=4, warmup_steps=1000, leak_rate=0.0):
        super().__init__()
        self.model = model
        self.num_buckets = num_buckets
        self.warmup_steps = warmup_steps
        self.leak_rate = leak_rate
        
        # PERSISTENCE: Register buffer so it's saved in state_dict
        self.register_buffer('current_step_count', torch.tensor(0, dtype=torch.long))
        
        self.metrics = {}
        
        # 1. Detect Layers
        self.layer_module_list = self._detect_transformer_layers(model)
        if not self.layer_module_list:
            raise ValueError("Could not automatically detect Transformer layer stack.")
        self.num_layers = len(self.layer_module_list)
        
        # 2. Configure Buckets and Routers
        self.routers = []
        self.bucket_stop_layers = []
        
        chunk_size = self.num_layers / num_buckets
        
        # Ensure we don't have duplicate cutoff layers if chunk_size is small
        created_router_layers = set()

        for b_idx in range(num_buckets):
            layers_to_keep = int(chunk_size * (num_buckets - b_idx))
            stop_layer = self.num_layers - layers_to_keep - 1
            
            self.bucket_stop_layers.append(stop_layer)
            
            if stop_layer >= 0:
                if stop_layer not in created_router_layers:
                    router = GradientRouter(stop_layer, leak_rate=leak_rate)
                    self.routers.append(router)
                    created_router_layers.add(stop_layer)
                    
                    target_module = self.layer_module_list[stop_layer]
                    self._register_forward_hook_for_router(target_module, router)

    def _register_forward_hook_for_router(self, module, router):
         def hook(mod, inp, out):
             if not self.training: return
             t = out[0] if isinstance(out, tuple) else out
             if isinstance(t, torch.Tensor):
                 router.register_hook(t)
         module.register_forward_hook(hook)

    def _detect_transformer_layers(self, model):
        largest_list = None
        max_len = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList):
                if len(module) > max_len:
                    max_len = len(module)
                    largest_list = module
        return largest_list

    def step(self):
        self.current_step_count += 1
        
    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        
        logits = None
        if isinstance(output, torch.Tensor): logits = output
        elif hasattr(output, 'logits'): logits = output.logits
        elif isinstance(output, tuple): logits = output[0]
        
        if logits is None or not self.training:
            return output
            
        with torch.no_grad():
            difficulty, conf, ent = compute_difficulty_score(logits)
            
            batch_size, seq_len = difficulty.shape
            flat_diff = difficulty.view(-1)
            
            # Rank-based bucketing
            ranks = torch.argsort(torch.argsort(flat_diff))
            num_tokens = flat_diff.numel()
            
            bucket_indices = (self.num_buckets - 1) - (ranks * self.num_buckets // num_tokens)
            bucket_indices = bucket_indices.clamp(0, self.num_buckets - 1)
            
            # Curriculum
            if self.warmup_steps > 0:
                progress = min(1.0, self.current_step_count.item() / self.warmup_steps)
                allowed_max_bucket = int(progress * (self.num_buckets - 1))
                bucket_indices = torch.min(bucket_indices, torch.tensor(allowed_max_bucket, device=difficulty.device))
            
            stop_layer_lookup = torch.tensor(self.bucket_stop_layers, device=difficulty.device)
            token_stop_layers = stop_layer_lookup[bucket_indices]
            token_stop_layers = token_stop_layers.view(batch_size, seq_len, 1)
            
            for router in self.routers:
                router.set_stop_layers(token_stop_layers)
                
            self.metrics = {
                'step': self.current_step_count.item(),
                'avg_bucket': bucket_indices.float().mean().item(),
                'avg_confidence': conf.mean().item()
            }

        return output
    
    def set_manual_stop_layers(self, stop_layers):
        for router in self.routers:
            router.set_stop_layers(stop_layers)
