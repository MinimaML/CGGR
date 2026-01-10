# --- EXTERNAL SCRIPT FOR CLEAN MEMORY ---
import time
import math
import re
import gc
import os
import sys
import torch
import torch.distributed as dist
import numpy as np
import argparse
from datetime import timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from muon import Muon, MuonWithAuxAdam
from cggr import CGGRModel, create_truncated_router
from huggingface_hub import create_repo, upload_folder
import wandb

# --- ⚙️ CONFIG ---
class RaceConfig:
    PROJECT_NAME = "cggr-vs-standard-race"
    RUN_DURATION_HOURS = 1.5
    SEED = 42
    ORG_NAME = "MinimaML"
    
    MODEL_ID = "google/gemma-3-4b-it"
    DATASET_ID = "AI-MO/NuminaMath-1.5"
    EVAL_DATASET = "HuggingFaceH4/aime_2024"
    MAX_SEQ_LEN = 2048
    
    # --- BATCH SIZES (PHYSICAL) ---
    BS_BASELINE = 1
    BS_CGGR = 4
    
    LR_MUON = 0.02
    LR_ADAM = 3e-4
    
    WARMUP_STEPS_BASE = 200
    WARMUP_STEPS_CGGR = 50
    EVAL_INTERVAL_MINS = 30

config = RaceConfig()

# --- 📦 DATA ---
def get_dataloaders(batch_size):
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
    ds_train = load_dataset(config.DATASET_ID, split="train", streaming=True)
    
    def collate_fn(examples):
        texts = [ex['problem'] + "\nSolution:\n" + ex['solution'] for ex in examples]
        enc = tokenizer(texts, padding="max_length", truncation=True, max_length=config.MAX_SEQ_LEN, return_tensors="pt")
        return enc['input_ids'].cuda(), enc['input_ids'].cuda()

    def data_generator():
        batch = []
        for example in ds_train:
            batch.append(example)
            if len(batch) == batch_size:
                yield collate_fn(batch)
                batch = []
    return data_generator(), tokenizer

try: ds_eval = load_dataset(config.EVAL_DATASET, split="train")
except: ds_eval = []

# --- 🧠 SHADOW ROUTING ---
class ShadowRouter:
    def __init__(self, model):
        self.router = create_truncated_router(model, num_layers=4).cuda()
        self.router.eval()
    
    def analyze(self, input_ids, logits, labels):
        if logits is None: return {}
        with torch.no_grad():
             loss_per_token = torch.nn.functional.cross_entropy(
                 logits.view(-1, logits.size(-1)), 
                 labels.view(-1), 
                 reduction='none'
             ).view(labels.shape)
             probs = torch.softmax(logits, dim=-1)
             entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
             k = int(entropy.numel() * 0.25)
             topk_thresh = torch.topk(entropy.view(-1), k).values[-1]
             mask_hard = entropy >= topk_thresh
             shadow_hard_loss = (loss_per_token * mask_hard).sum() / (mask_hard.sum() + 1e-9)
             return {
                 "shadow/hard_loss": shadow_hard_loss.item(),
                 "shadow/entropy_mean": entropy.mean().item()
             }

# --- 🧠 SETUP ---
def setup_training(mode="standard"):
    print(f"Loading {config.MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID, 
        dtype=torch.bfloat16
    )
    model.config.use_cache = False
    model = model.cuda()
    model.gradient_checkpointing_enable()

    shadow_router = None
    if mode == "cggr":
        print("Initializing CGGR...")
        router = create_truncated_router(model, num_layers=4).cuda()
        model = CGGRModel(model, router=router, min_tokens_ratio=0.25, warmup_steps=config.WARMUP_STEPS_CGGR)
        base_model = model.model
    else:
        print("Initializing Shadow Router...")
        shadow_router = ShadowRouter(model)
        base_model = model

    print("Configuring MuonWithAuxAdam...")
    try:
        text_model = base_model.model.language_model
        base_model.body = text_model
        base_model.head = base_model.lm_head
        base_model.embed = text_model.embed_tokens
    except AttributeError as e:
        print(f"⚠️ Mapping failed: {e}. Attempting recovery...")
        base_model.body = getattr(base_model, 'model', base_model)
        base_model.head = getattr(base_model, 'lm_head', None)
        base_model.embed = getattr(base_model.body, 'embed_tokens', None)
    
    hidden_weights = []
    hidden_gains_biases = []
    nonhidden_params = []
    seen_ids = set()
    
    for p in list(base_model.head.parameters()) + list(base_model.embed.parameters()):
        if id(p) not in seen_ids:
            nonhidden_params.append(p); seen_ids.add(id(p))
    if mode == "cggr":
        for p in model.router.parameters():
            if id(p) not in seen_ids:
                nonhidden_params.append(p); seen_ids.add(id(p))
    for p in base_model.body.parameters():
        if id(p) in seen_ids or not p.requires_grad: continue
        if p.ndim >= 2: hidden_weights.append(p)
        else: hidden_gains_biases.append(p)
        seen_ids.add(id(p))

    print(f"📦 Hidden Weights: {len(hidden_weights)} | Hidden Gains: {len(hidden_gains_biases)} | NonHidden: {len(nonhidden_params)}")

    param_groups = [
        dict(params=hidden_weights, use_muon=True, lr=config.LR_MUON, weight_decay=0.01),
        dict(params=hidden_gains_biases + nonhidden_params, use_muon=False, 
             lr=config.LR_ADAM, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8),
    ]
    optimizer = MuonWithAuxAdam(param_groups)
    return model, optimizer, shadow_router

# --- 🕵️ ROBUST EVAL ---
def extract_answer(text):
    boxed_match = re.search(r'\\boxed\{(.*?)\}', text)
    if boxed_match: return boxed_match.group(1).strip()
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers: return numbers[-1]
    return ""

def evaluate_and_upload(model, tokenizer, run_name, hour_idx, dry_run=False):
    num_problems = 20 if dry_run else len(ds_eval)
    print(f"🕒 Evaluation Hour {hour_idx} ({num_problems} problems)... ")
    model.eval()
    base_gen_model = model.model if isinstance(model, CGGRModel) else model
    PROMPT_TEMPLATE = "Problem: {problem}\nLet's think step by step to solve this.\nSolution:"
    correct, total = 0, 0
    dataset_iter = list(ds_eval)[:num_problems]

    for ex in dataset_iter:
        prompt = PROMPT_TEMPLATE.format(problem=ex['problem'])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = base_gen_model.generate(**inputs, max_new_tokens=1024, temperature=0.7, do_sample=True)
        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred, truth = extract_answer(gen_text), str(ex.get('answer', '')).strip()
        if pred and truth:
             p_c, t_c = pred.replace(',', ''), truth.replace(',', '')
             try: 
                 if abs(float(p_c) - float(t_c)) < 1e-6: correct += 1
             except: 
                 if pred.lower() == truth.lower(): correct += 1
        total += 1
    
    acc = correct / total if total > 0 else 0
    print(f"📈 AIME Score: {acc:.4f}")
    
    if not dry_run:
         repo_id = f"{config.ORG_NAME}/{run_name}-hr{hour_idx}"
         local_path = f"./tmp_checkpoints/{run_name}/hour_{hour_idx}"
         model.save_pretrained(local_path); tokenizer.save_pretrained(local_path)
         try:
             create_repo(repo_id, exist_ok=True); upload_folder(folder_path=local_path, repo_id=repo_id, repo_type="model")
         except Exception as e: print(f"❌ Upload Failed: {e}")
    
    model.train()
    return acc

# --- 🏁 MAIN LOOP ---
class RaceTimer:
    def __init__(self, limit_hours):
        self.limit_seconds = limit_hours * 3600
        self.start_time = time.time()
        self.paused_duration = 0
        self.last_pause_start = None
        self.is_paused = False
    def pause(self):
        if not self.is_paused: self.last_pause_start = time.time(); self.is_paused = True
    def resume(self):
        if self.is_paused: self.paused_duration += time.time() - self.last_pause_start; self.is_paused = False
    def elapsed_training(self):
        curr = time.time() if not self.is_paused else self.last_pause_start
        return (curr - self.start_time) - self.paused_duration
    def check_limit(self): return self.elapsed_training() < self.limit_seconds

def set_seed(seed=42):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)

if __name__ == '__main__':
    # --- DUMMY DIST INIT (Required for Muon) ---
    if not dist.is_initialized():
        print("Initializing Dummy Process Group for Muon...")
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    
    mode, dry_run = args.mode, args.dry_run
    set_seed(config.SEED)
    
    run_name = f"{config.PROJECT_NAME}_{mode}_robust"
    bs = config.BS_CGGR if mode == "cggr" else config.BS_BASELINE
    
    wandb.init(project=config.PROJECT_NAME, name=run_name, config={"mode": mode, "bs": bs})
    
    train_iter, tokenizer = get_dataloaders(bs)
    model, optimizer, shadow_router = setup_training(mode)
    
    timer = RaceTimer(limit_hours=0.02 if dry_run else config.RUN_DURATION_HOURS)
    next_check, hour_counter, step = config.EVAL_INTERVAL_MINS * 60, 1, 0

    print(f"🚦 STARTING {mode.upper()} RACE (BS={bs})")
    
    while timer.check_limit():
        step_start = time.time()
        # Eval check
        if timer.elapsed_training() >= next_check:
            timer.pause()
            acc = evaluate_and_upload(model, tokenizer, run_name, hour_counter, dry_run)
            wandb.log({"eval/aime_pass1": acc, "eval/hour": hour_counter})
            hour_counter += 1
            next_check += (config.EVAL_INTERVAL_MINS * 60)
            timer.resume()
            if dry_run: break

        try: batch_ids, batch_labels = next(train_iter)
        except: print("Data exhausted"); break
        
        optimizer.zero_grad()
        outputs = model(batch_ids, labels=batch_labels)
        loss = outputs if isinstance(outputs, torch.Tensor) else outputs.loss
        
        if mode == 'standard' and shadow_router:
             wandb.log(shadow_router.analyze(batch_ids, outputs.logits, batch_labels), commit=False)
        
        loss.backward()
        optimizer.step()
        if hasattr(model, 'step'): model.step()
        
        if step % 10 == 0:
            dt = time.time() - step_start
            tps = (bs * config.MAX_SEQ_LEN) / dt if dt > 0 else 0
            wandb.log({"train/loss": loss.item(), "train/tps": tps, "time/mins": timer.elapsed_training()/60})
            print(f"Step {step} | Loss: {loss.item():.4f} | TPS: {tps:.0f}")
        step += 1
        
    print("🏁 FINISH")
    evaluate_and_upload(model, tokenizer, run_name, "final", dry_run)
    wandb.finish()
