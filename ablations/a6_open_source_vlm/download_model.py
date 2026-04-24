"""
download_model.py — Download Gemma 4 31B Dense IT to data/models/gemma-4-31b-it

Usage:
    python download_model.py

Requires HF_TOKEN env var or ~/.cache/huggingface/token to be set.
Run once before the evaluator.
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))

MODEL_ID   = "google/gemma-4-31B-it"
LOCAL_DIR  = DATA_ROOT / "models/gemma-4-31b-it"

def main():
    # Gemma 4 is Apache 2.0 licensed and NOT gated — no HF token required.
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_ID} → {LOCAL_DIR}")
    print("This will download ~63GB (bf16 safetensors). Progress shown below.\n")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(LOCAL_DIR),
        ignore_patterns=["*.pt", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"\nDone. Model saved to {LOCAL_DIR}")

if __name__ == "__main__":
    main()
