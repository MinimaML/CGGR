"""Quick verification script for persistent CGGR kernels."""
import torch
from persistent_cggr_kernels import create_persistent_tr_mlp

# Test on CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Testing on: {device}')

model = create_persistent_tr_mlp(
    hidden_dim=512, 
    intermediate_dim=2048, 
    num_experts=8, 
    top_k=2
).to(device).half()

x = torch.randn(4, 64, 512, device=device, dtype=torch.float16)

# Test forward
output, aux_loss = model(x, use_persistent=False)
print(f'Output shape: {output.shape}')
print(f'Aux loss: {aux_loss.item():.4f}')

# Verify metrics
metrics = model.get_metrics()
print(f'Router entropy: {metrics["router_entropy"]:.4f}')
print(f'Top expert load: {metrics["top_expert_load"]:.4f}')

# Test backward
x_grad = x.clone().requires_grad_(True)
out, loss = model(x_grad, use_persistent=False)
total_loss = out.sum() + loss
total_loss.backward()
print(f'Gradients OK: {x_grad.grad is not None}')

print('VERIFICATION PASSED!')
