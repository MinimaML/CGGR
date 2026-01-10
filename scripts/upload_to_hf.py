import argparse
from huggingface_hub import HfApi, create_repo
import os
from pathlib import Path

# --- CONFIGURATION ---
MODEL_PATH = r"c:\Users\wrc02\Desktop\CGGR\results\math_finetune\cggr_6.0h\final_model"
ORG_NAME = "MinimaML"
REPO_NAME = "SmolLM-135M-CGGR-Math"

def upload_to_hub(token):
    api = HfApi()
    repo_id = f"{ORG_NAME}/{REPO_NAME}"
    
    print(f"Checking repository: {repo_id}")
    try:
        create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")

    print(f"Uploading all model assets from {MODEL_PATH}...")
    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Initial upload of CGGR-specialized Math model"
    )
    
    print(f"\n✨ SUCCESS! Your model is live at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload CGGR model to Hugging Face Hub")
    parser.add_argument("--token", required=True, help="HF Write Token (hf_...)")
    args = parser.parse_args()
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path {MODEL_PATH} not found.")
    else:
        upload_to_hub(args.token)
