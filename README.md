# Confidence-Gated Gradient Routing (CGGR)

CGGR is a PyTorch library that implements Multi-Stage Gradient Routing for Transformer models. It selectively stops gradient backpropagation for tokens based on their confidence and entropy, effectively reducing the compute required for the backward pass.

## Mechanism

### Gradient Routing
The library identifies tokens as "easy" or "hard" based on a difficulty score derived from the model's output logits.
$$D_t = \text{Entropy}(P_t) - \text{Confidence}(P_t)$$

- **Hard Tokens**: Gradients propagate through all layers.
- **Easy Tokens**: Gradients are blocked at earlier layers (e.g., top 25% or 50% of the model).

This is achieved by registering backward hooks on specific layers that multiply gradients by a binary mask (0 or 1).

### Multi-Stage Bucketing
Tokens are sorted by difficulty and assigned to "buckets" which correspond to different gradient depths.

| Bucket | Difficulty  | Gradient Depth    |
| :----- | :---------- | :---------------- |
| **0**  | Highest     | 100% (All Layers) |
| **1**  | Medium-High | 75%               |
| **2**  | Medium-Low  | 50%               |
| **3**  | Lowest      | 25%               |

### Curriculum Learning
To ensure training stability, the system uses a linear warmup. Initially, all tokens are processed at full depth. Over a specified number of `warmup_steps`, the system gradually enables the shallower gradient paths.

## Usage

The `CGGRWrapper` class wraps an existing PyTorch module. It automatically detects the Transformer layer stack and injects the necessary hooks.

```python
import torch
from cggr import CGGRWrapper
from my_model import MyTransformer

# 1. Initialize your standard model
model = MyTransformer(...)

# 2. Wrap with CGGR
# num_buckets: Number of depth tiers (e.g., 4)
# warmup_steps: Number of steps to reach target efficiency
model = CGGRWrapper(model, num_buckets=4, warmup_steps=1000)

# 3. Training Loop
optimizer = torch.optim.Adam(model.parameters())

for batch in dataloader:
    # Forward pass
    logits = model(input_ids) 
    loss = criterion(logits, targets)
    
    # Backward pass with dynamic routing
    loss.backward() 
    
    optimizer.step()
    
    # Update curriculum state
    model.step()
```

## Configuration Options

- **`num_buckets`**: Number of efficiency tiers.
- **`warmup_steps`**: Length of the curriculum warmup phase.
- **`leak_rate`**: (Optional) Allows a fraction of gradients to pass through stopped paths. Default is 0.0.

## Persistence
The wrapper registers a buffer for the step count, so the curriculum state is automatically saved and loaded with `model.state_dict()`.

## Compatibility

CGGR is designed for **Transformer-based LLMs** (Llama, Mistral, GPT), but it was **specifically engineered to synergize with SRDE (Sparse Routed Delta Experts)**.

*   **SRDE Optimization**: When combined with SRDE, CGGR enables "Double Sparsity"—sparsifying both the forward pass (via MoE routing) and the backward pass (via Gradient Routing). This combination yields the theoretical maximum training efficiency.
*   **Dense Models**: Works out of the box for standard Transformers.
