import torch
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights

MODEL_ID = "google/gemma-3-4b-it"
config = AutoConfig.from_pretrained(MODEL_ID)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

lm = model.model.language_model
print(f"--- lm (type: {type(lm)}) Children ---")
for name, _ in lm.named_children():
    print(f" - {name}")
