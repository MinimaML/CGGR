import torch
import torch.nn as nn
import torch.optim as optim
# Since we moved files, we need to adjust imports if running from examples/
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.simple_transformer import SimpleTransformer
from cggr import CGGRWrapper
import os

def run_test():
    print("--- Setting up Final Polish Test ---")
    vocab_size = 100
    d_model = 32
    nhead = 4
    num_layers = 12
    NUM_BUCKETS = 2 
    
    # 1. Test Persistence
    base_model = SimpleTransformer(vocab_size, d_model, nhead, num_layers)
    model = CGGRWrapper(base_model, num_buckets=NUM_BUCKETS, warmup_steps=100)
    
    # Simulate training steps
    model.current_step_count.fill_(50)
    print(f"Current Step before save: {model.current_step_count.item()}")
    
    # Save State
    torch.save(model.state_dict(), "ckpt.pt")
    
    # New Model (Reset)
    model_new = CGGRWrapper(SimpleTransformer(vocab_size, d_model, nhead, num_layers), 
                            num_buckets=NUM_BUCKETS, warmup_steps=100)
    print(f"New Model Step before load: {model_new.current_step_count.item()}")
    
    # Load
    model_new.load_state_dict(torch.load("ckpt.pt"))
    print(f"New Model Step after load: {model_new.current_step_count.item()}")
    
    if model_new.current_step_count.item() == 50:
        print("SUCCESS: State Persistence working.")
    else:
        print("FAILURE: State Persistence failed.")

    # 2. Test Leaky Gating
    print("\n--- Verifying Leaky Gating ---")
    LEAK_RATE = 0.1
    model_leak = CGGRWrapper(SimpleTransformer(vocab_size, d_model, nhead, num_layers), 
                             num_buckets=2, warmup_steps=0, leak_rate=LEAK_RATE)
    model_leak.train()
    
    # Hook Embeddings
    feature_grads = {}
    def get_hook(name):
        def hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple): g = grad_output[0]
            else: g = grad_output
            feature_grads[name] = g
        return hook
    model_leak.model.embedding.register_backward_hook(get_hook("embed"))
    
    x = torch.randint(0, vocab_size, (2, 5))
    targets = torch.randint(0, vocab_size, (2, 5))
    logits, _ = model_leak(x)
    loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), targets.view(-1))
    
    # Force Stop Layer to be deep, so gradients SHOULD be blocked at embedding
    # We have 12 layers. Bucket 1 (Easy) stops at let's say 5.
    # We force stop layer to be 5.
    # Embedding is at -1. 
    # Logic: If stop_layer (5) < current (Embed/0? No).
    # The routers are at the STOP layers.
    # So valid gradients pass if stop_layer >= current_router_layer? No.
    # Router logic: mask = (stop_layer_tensor < self.layer_idx) (Pass if stop is deeper than router)
    # If stop=5, and router is at 11 (Top), 5 < 11 True. Pass.
    # If router is at 5. 5 < 5 False. Block!
    # So Router at 5 blocks gradients coming from 6.
    # Implies gradients below 5 (4,3,2,1,0) should be ZERO.
    
    # BUT with LEAK RATE 0.1, they should be approx 0.1 * Full Grad?
    # Hard to reference "Full Grad".
    # Just check if > 0.
    
    # Let's force stop layer to 0.
    manual_stops = torch.zeros(2, 5, 1) # Stop at layer 0
    model_leak.set_manual_stop_layers(manual_stops)
    
    model_leak.zero_grad()
    loss.backward()
    
    grad_norm = feature_grads["embed"].norm()
    print(f"Leaky Gradient Norm (Input): {grad_norm.item():.6f}")
    
    if grad_norm.item() > 1e-5:
        print("SUCCESS: Leaky gating allows gradient flow.")
    else:
        print("FAILURE: Leaky gating blocked gradients completely (or gradients were naturally 0).")
        
    # Clean up
    if os.path.exists("ckpt.pt"): os.remove("ckpt.pt")

if __name__ == "__main__":
    run_test()
