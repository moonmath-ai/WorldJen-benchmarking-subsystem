"""
Compute semantic adherence scores for all 50 prompts.

For each (model, prompt) pair, embeds the enhanced prompt text and the
corresponding video using the Gemini Embedding API, then records the
cosine similarity as the semantic adherence score.

Usage:
    python semantic_adherence/compute_semantic_adherence.py

    # Custom paths:
    python semantic_adherence/compute_semantic_adherence.py \
        --prompts data/prompts/prompts_50.jsonl \
        --video-base data/videos \
        --output data/results/summaries/semantic_adherence_results_50.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

EMBEDDING_MODEL = "models/gemini-embedding-2-preview"

MODEL_DIRS = {
    "fal-ai_hunyuan-video-v1.5":   "fal-ai_hunyuan-video-v1.5_text-to-video",
    "fal-ai_kling-video_v2.6_pro": "fal-ai_kling-video_v2.6_pro_text-to-video",
    "fal-ai_ltx-2":                "fal-ai_ltx-2_text-to-video",
    "fal-ai_veo3.1_fast":          "fal-ai_veo3.1_fast",
    "fal-ai_wan_v2.2-a14b":        "fal-ai_wan_v2.2-a14b_text-to-video",
    "wan2.1-1.3b":                 "wan2.1-1.3b",
}


def parse_args():
    p = argparse.ArgumentParser(description="Compute Gemini-embedding semantic adherence scores.")
    p.add_argument("--prompts", default="data/prompts/prompts_50.jsonl",
                   help="Path to prompts JSONL (enhanced_prompt + prompt_id fields).")
    p.add_argument("--video-base", default="data/videos",
                   help="Base directory containing one sub-folder per model.")
    p.add_argument("--output",
                   default=str(Path(os.environ.get("WORLDJEN_DATA_ROOT", Path(__file__).resolve().parent.parent / "data")) / "results/summaries/semantic_adherence_results_50.json"),
                   help="Output JSON file path.")
    return p.parse_args()


def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def load_prompts(path: Path) -> dict:
    prompts = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            prompts[d["prompt_id"]] = d["enhanced_prompt"]
    return prompts


def get_embedding(content, retries=3):
    for attempt in range(retries):
        try:
            result = genai.embed_content(model=EMBEDDING_MODEL, content=content)
            return result["embedding"]
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = 2 ** attempt * 5
                print(f"  Rate limit — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Embedding error: {e}")
                return None
    return None


def main():
    args = parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Add it to a .env file or set it as an environment variable."
        )
    genai.configure(api_key=api_key)

    prompts = load_prompts(Path(args.prompts))
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        print(f"Resuming — {sum(len(v) for v in results.values())} entries already saved.")
    else:
        results = {}

    prompt_emb_cache = {}
    video_base = Path(args.video_base)

    for model_name, subdir in MODEL_DIRS.items():
        existing = results.get(model_name, [])
        if len(existing) >= len(prompts):
            print(f"Skipping {model_name} (already complete: {len(existing)} entries).")
            continue

        model_dir = video_base / subdir
        if not model_dir.is_dir():
            print(f"Warning: video directory not found: {model_dir} — skipping.")
            continue

        print(f"\nProcessing model: {model_name}")
        if model_name not in results:
            results[model_name] = []

        processed_ids = {r["prompt_id"] for r in results[model_name]}
        video_files = sorted(model_dir.glob("prompt_*.mp4"))
        print(f"  Found {len(video_files)} videos.")

        for video_path in video_files:
            prompt_id = video_path.stem
            if prompt_id in processed_ids:
                continue
            if prompt_id not in prompts:
                print(f"  Skipping {video_path.name}: prompt_id not in prompts file.")
                continue

            prompt_text = prompts[prompt_id]

            if prompt_id not in prompt_emb_cache:
                print(f"  Embedding prompt {prompt_id}...")
                emb = get_embedding(prompt_text)
                if emb:
                    prompt_emb_cache[prompt_id] = emb
                else:
                    continue
            prompt_emb = prompt_emb_cache[prompt_id]

            print(f"  Uploading video {video_path.name}...")
            try:
                video_file = genai.upload_file(path=str(video_path))
                video_emb = get_embedding(video_file)
                if video_emb:
                    score = cosine_similarity(prompt_emb, video_emb)
                    print(f"    Score: {score:.4f}")
                    results[model_name].append({"prompt_id": prompt_id, "score": score})
                else:
                    print(f"    Failed to embed {video_path.name}")
            except Exception as e:
                print(f"    Error processing {video_path.name}: {e}")

        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"  Progress saved for {model_name}")

    print(f"\nDone. Results written to {output_path}")


if __name__ == "__main__":
    main()
