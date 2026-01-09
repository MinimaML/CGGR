import torch
from transformers import AutoModelForCausalLM, AutoConfig
import os

MODEL_ID = "google/gemma-3-4b-it"

print(f"Inspecting {MODEL_ID}...")
config = AutoConfig.from_pretrained(MODEL_ID)

# Use meta device to avoid downloading/loading full weights just to see names
from accelerate import init_empty_weights
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

print("\n--- Root Children ---")
for name, _ in model.named_children():
    print(f" - {name}")

if hasattr(model, 'model'):
    print("\n--- model.model (backbone) Children ---")
    for name, _ in model.model.named_children():
        print(f" - {name}")
