import torch
from transformers import AutoModelForCausalLM, AutoConfig

MODEL_ID = "google/gemma-3-4b-it"

print(f"Inspecting {MODEL_ID}...")
config = AutoConfig.from_pretrained(MODEL_ID)
print(f"Config: {config}")

# We don't need to load the full weights to see names, just init with meta or small
model = AutoModelForCausalLM.from_config(config)

print("\n--- Parameter Names ---")
for name, p in model.named_parameters():
    print(f"{name} | {p.ndim}D | {p.shape}")

print("\n--- Module structure ---")
for name, module in model.named_children():
    print(f"Child: {name} ({type(module)})")
