"""
vlm_evaluator_gemma4.py — WorldJen VLM evaluation using Gemma 4 31B Dense IT.

Architecture:
  - 8 worker processes, one per GPU (CUDA_VISIBLE_DEVICES=N)
  - Each worker loads its own model instance in bf16 on a single H200
  - Work queue: (model_path, prompt_id) pairs distributed across workers
  - Output format identical to parallel_vlm_evaluator.py (Gemini) for drop-in
    compatibility with unified_analyzer.py

Usage:
    # First download the model (run once):
    python download_model.py

    # Then run the evaluator:
    python vlm_evaluator_gemma4.py

    # Dry run (no GPU, just check paths/questions):
    python vlm_evaluator_gemma4.py --dry-run

    # Resume (skips already-completed output files):
    python vlm_evaluator_gemma4.py           # auto-resume by default

    # Limit to specific models or prompts:
    python vlm_evaluator_gemma4.py --models fal-ai_veo3.1_fast wan2.1-1.3b
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
import datetime
import traceback
from pathlib import Path

import cv2
import PIL.Image
import numpy as np
import torch

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_PATH   = Path("/data/models/gemma-4-31b-it")
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))
VIDEO_ROOT   = DATA_ROOT / "videos"
PROMPTS_FILE = DATA_ROOT / "prompts/prompts_50.jsonl"
VQA_FILE     = DATA_ROOT / "prompts/vqa_questions_50prompts.jsonl"
OUTPUT_DIR   = DATA_ROOT / "results/gemma4_vlm"
N_GPUS       = 8
MAX_RETRIES  = 3
GEMMA_MODEL_TAG = "google/gemma-4-31B-it"   # recorded in output JSON

MODELS_PATHS = [
    "fal-ai_kling-video_v2.6_pro_text-to-video",
    "fal-ai_ltx-2_text-to-video",
    "fal-ai_hunyuan-video-v1.5_text-to-video",
    "fal-ai_veo3.1_fast",
    "fal-ai_wan_v2.2-a14b_text-to-video",
    "wan2.1-1.3b",
]

DIMENSION_MODES = {
    "semantic_adherence":   "holistic",
    "composition_framing":  "holistic",
    "aesthetic_quality":    "holistic",
    "lighting_volumetric":  "holistic",
    "color_harmony":        "holistic",
    "structural_gestalt":   "holistic",
    "dynamic_degree":       "holistic",
    "semantic_drift":       "holistic",
    "scene_consistency":    "sampled",
    "object_permanence":    "sampled",
    "subject_consistency":  "sampled",
    "motion_smoothness":    "micro",
    "temporal_flickering":  "micro",
    "inertial_consistency": "micro",
    "physical_mechanics":   "micro",
    "human_fidelity":       "micro",
}

# System prompt — identical intent to the Gemini evaluator prompt
SYSTEM_PROMPT = None  # No separate system prompt — matches Gemini evaluator style


# ── Frame extraction (identical to Gemini evaluator) ───────────────────────────

def extract_frames(video_path: Path, mode: str = "sampled", num_frames: int = 16) -> list:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if mode == "sampled":
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    elif mode == "micro":
        # First ~2s at high frequency (every 5th frame, up to 60)
        indices = list(range(0, min(total_frames, 60), 5))
    else:  # holistic
        indices = [int(i * total_frames / 32) for i in range(min(32, total_frames))]
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(PIL.Image.fromarray(frame_rgb))
    cap.release()
    return frames


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(dimension: str, questions: list, original_prompt: str) -> str:
    """Matches the Gemini evaluator prompt structure exactly."""
    prompt = f"You are evaluating a video generated from a prompt. Dimension: {dimension}\n\nQuestions to answer:\n"
    for i, q in enumerate(questions, 1):
        prompt += f"{i}. {q['question']}\n   Rubric: {q['rubric_description']}\n"
    prompt += "\nAnswer each question with a score (1-5) and a SHORT justification (max 15 words). Return ONLY a JSON list of objects: [{\"score\": X, \"justification\": \"...\"}]"
    return prompt


REFUSAL_PHRASES = [
    "i cannot", "i can't", "cannot answer", "not applicable",
    "no physical", "no humans", "not visible", "unable to assess",
    "cannot evaluate", "not present", "n/a",
]

def parse_response(text: str, n_questions: int) -> list | None:
    """Extract JSON list from model output, tolerating markdown fences.
    Returns None on parse failure. Returns a list of null-score dicts on refusal
    so the dimension is recorded as unanswerable rather than retried 3 times.
    """
    stripped = text.strip().lower()

    # Detect refusals early — return sentinel so caller skips retries
    if any(p in stripped for p in REFUSAL_PHRASES) and "[" not in stripped:
        return "REFUSAL"

    # Strip markdown fences if present
    text = text.strip()
    for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            text = m.group(1).strip()
            break
    # Find first [...] block; if truncated, try to close it
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        text = m.group(0)
    elif "[" in text:
        # Truncated — attempt to close the array and salvage complete objects
        text = text[text.index("["):]
        text = re.sub(r',\s*\{[^}]*$', '', text)  # remove last incomplete object
        text = text.rstrip(", \n") + "]"
    try:
        result = json.loads(text)
        if isinstance(result, list) and len(result) == n_questions:
            cleaned = []
            for item in result:
                score = int(item.get("score", 0))
                if not (1 <= score <= 5):
                    return None
                cleaned.append({
                    "score": score,
                    "justification": str(item.get("justification", ""))
                })
            return cleaned
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return None


# ── Per-GPU worker ─────────────────────────────────────────────────────────────

def worker_main(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue,
                dry_run: bool, prompt_text_map: dict):
    """
    Runs in a separate process. Loads model on gpu_id, then processes tasks
    from task_queue until it receives None (sentinel).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if not dry_run:
        # Lazy import inside worker so the parent process doesn't load CUDA
        from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer

        print(f"[GPU {gpu_id}] Loading model from {MODEL_PATH} ...", flush=True)
        t0 = time.time()

        processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",   # CUDA_VISIBLE_DEVICES already set to gpu_id
            attn_implementation="eager",  # flash_attention_2 may not be installed
        )
        model.eval()
        print(f"[GPU {gpu_id}] Model loaded in {time.time()-t0:.1f}s", flush=True)
    else:
        processor = model = None
        print(f"[GPU {gpu_id}] DRY RUN — skipping model load", flush=True)

    while True:
        task = task_queue.get()
        if task is None:
            break  # sentinel

        model_path, prompt_id, vqa, output_file, prompt_set = task
        model_name = Path(model_path).name
        original_prompt = prompt_text_map.get(prompt_id, "")

        print(f"[GPU {gpu_id}] START {model_name}/{prompt_id}", flush=True)
        t_video = time.time()

        if dry_run:
            result_queue.put(("ok", model_name, prompt_id))
            continue

        # Load partial results from a previous interrupted run if present
        partial_file = output_file.with_suffix(".partial.json")
        dim_results = {}
        if partial_file.exists():
            try:
                saved = json.loads(partial_file.read_text())
                dim_results = saved.get("results", {})
                n_done = sum(1 for v in dim_results.values() if v is not None)
                print(f"[GPU {gpu_id}]   Resuming from partial ({n_done} dims already done)", flush=True)
            except Exception:
                dim_results = {}

        for dimension, questions in vqa.items():
            if dimension in dim_results:
                print(f"[GPU {gpu_id}]   SKIP {dimension} (already done)", flush=True)
                continue
            mode = DIMENSION_MODES.get(dimension, "holistic")
            video_path = VIDEO_ROOT / model_path / f"{prompt_id}.mp4"
            if not video_path.exists():
                print(f"[GPU {gpu_id}]   MISSING video: {video_path}", flush=True)
                dim_results[dimension] = None
                continue

            frames = extract_frames(video_path, mode)
            if not frames:
                print(f"[GPU {gpu_id}]   No frames extracted for {dimension}", flush=True)
                dim_results[dimension] = None
                continue

            user_text = build_prompt(dimension, questions, original_prompt)

            # Matches Gemini evaluator: prompt text + frames, no system turn
            messages = [
                {
                    "role": "user",
                    "content": (
                        [{"type": "image", "image": img} for img in frames]
                        + [{"type": "text", "text": user_text}]
                    ),
                },
            ]

            for attempt in range(MAX_RETRIES):
                try:
                    # Apply chat template
                    inputs = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to("cuda:0", dtype=torch.bfloat16)

                    with torch.inference_mode():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=2048,
                            do_sample=False,          # greedy — deterministic
                            temperature=None,
                            top_p=None,
                            top_k=None,               # suppress invalid-flag warning
                        )

                    # Decode only the newly generated tokens
                    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
                    text = processor.decode(new_tokens, skip_special_tokens=True)

                    parsed = parse_response(text, len(questions))
                    if parsed == "REFUSAL":
                        print(f"[GPU {gpu_id}]   REFUSAL {dimension} (not applicable to this video) → null", flush=True)
                        dim_results[dimension] = None
                        partial_file.write_text(json.dumps({"results": dim_results}, indent=2))
                        break
                    elif parsed is not None:
                        dim_results[dimension] = parsed
                        print(f"[GPU {gpu_id}]   OK  {dimension} ({len(frames)} frames)", flush=True)
                        # Save partial progress so a crash can be resumed
                        partial_file.write_text(json.dumps({"results": dim_results}, indent=2))
                        break
                    else:
                        print(f"[GPU {gpu_id}]   PARSE FAIL {dimension} attempt {attempt+1}: {text[:120]}", flush=True)
                        if attempt == MAX_RETRIES - 1:
                            dim_results[dimension] = None

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"[GPU {gpu_id}]   OOM on {dimension} attempt {attempt+1} — clearing cache", flush=True)
                    time.sleep(2)
                except Exception as e:
                    print(f"[GPU {gpu_id}]   ERROR {dimension} attempt {attempt+1}: {e}", flush=True)
                    traceback.print_exc()
                    time.sleep(3)
                    if attempt == MAX_RETRIES - 1:
                        dim_results[dimension] = None

        # Save result
        out = {
            "model": model_name,
            "prompt_id": prompt_id,
            "prompt_set": prompt_set,
            "gemini_model": GEMMA_MODEL_TAG,   # field name kept for analyser compat
            "eval_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "results": dim_results,
        }
        with open(output_file, "w") as f:
            json.dump(out, f, indent=2)

        # Clean up partial file now that the full result is saved
        if partial_file.exists():
            partial_file.unlink()

        elapsed = time.time() - t_video
        n_ok = sum(1 for v in dim_results.values() if v is not None)
        print(f"[GPU {gpu_id}] DONE {model_name}/{prompt_id} "
              f"({n_ok}/{len(dim_results)} dims OK, {elapsed:.1f}s) → {output_file}", flush=True)
        result_queue.put(("ok", model_name, prompt_id))

    print(f"[GPU {gpu_id}] Worker exiting.", flush=True)


# ── Main orchestrator ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WorldJen Gemma 4 VLM Evaluator")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip model loading and inference — just check paths")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Restrict to specific model folder names")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="Restrict to specific prompt IDs")
    parser.add_argument("--gpus", type=int, default=N_GPUS,
                        help=f"Number of GPUs to use (default: {N_GPUS})")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate model path
    if not args.dry_run and not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run:  python download_model.py")
        sys.exit(1)

    # Load prompts
    prompt_set_map = {}
    prompt_text_map = {}
    with open(PROMPTS_FILE) as f:
        for line in f:
            e = json.loads(line)
            prompt_set_map[e["prompt_id"]] = e.get("prompt_set", "unknown")
            prompt_text_map[e["prompt_id"]] = e.get("enhanced_prompt", e.get("prompt", ""))

    # Load VQA questions
    questions_map = {}
    with open(VQA_FILE) as f:
        for line in f:
            e = json.loads(line)
            questions_map[e["prompt_id"]] = e["vqa"]

    # Build task list
    model_filter = set(args.models) if args.models else None
    prompt_filter = set(args.prompts) if args.prompts else None

    tasks = []
    for rel_path in MODELS_PATHS:
        model_name = Path(rel_path).name
        if model_filter and model_name not in model_filter and rel_path not in model_filter:
            continue
        video_dir = VIDEO_ROOT / rel_path
        if not video_dir.exists():
            print(f"WARNING: video dir not found: {video_dir}")
            continue
        for video_file in sorted(video_dir.glob("*.mp4")):
            pid = video_file.stem
            if prompt_filter and pid not in prompt_filter:
                continue
            if pid not in questions_map:
                continue
            output_file = OUTPUT_DIR / f"{model_name}_{pid}.json"
            if output_file.exists():
                continue  # auto-resume: skip completed
            tasks.append((
                rel_path,
                pid,
                questions_map[pid],
                output_file,
                prompt_set_map.get(pid, "unknown"),
            ))

    n_total = len(tasks)
    if n_total == 0:
        print("No tasks to run (all output files already exist). Done.")
        return

    n_gpus = min(args.gpus, N_GPUS, n_total)
    print(f"WorldJen Gemma 4 Evaluator")
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Tasks  : {n_total} videos × 16 dims = ~{n_total*16} VLM calls")
    print(f"  Workers: {n_gpus} GPUs")
    print(f"  Output : {OUTPUT_DIR}")
    print()

    # Use 'spawn' to avoid CUDA fork issues
    mp.set_start_method("spawn", force=True)
    task_queue   = mp.Queue()
    result_queue = mp.Queue()

    # Enqueue all tasks
    for task in tasks:
        task_queue.put(task)
    # Enqueue sentinels (one per worker)
    for _ in range(n_gpus):
        task_queue.put(None)

    # Launch workers
    workers = []
    for gpu_id in range(n_gpus):
        p = mp.Process(
            target=worker_main,
            args=(gpu_id, task_queue, result_queue, args.dry_run, prompt_text_map),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # Collect results with progress
    t_start = time.time()
    completed = 0
    while completed < n_total:
        status, model_name, prompt_id = result_queue.get()
        completed += 1
        elapsed = time.time() - t_start
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (n_total - completed) / rate if rate > 0 else 0
        print(f"Progress: {completed}/{n_total}  "
              f"({100*completed/n_total:.1f}%)  "
              f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m",
              flush=True)

    for p in workers:
        p.join()

    total_time = time.time() - t_start
    print(f"\nAll done. {n_total} videos evaluated in {total_time/60:.1f} minutes.")
    print(f"Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
