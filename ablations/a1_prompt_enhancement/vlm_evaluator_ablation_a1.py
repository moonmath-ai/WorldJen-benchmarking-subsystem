"""
vlm_evaluator_ablation_a1.py

VLM evaluation for the A1 ablation set (20 UNENHANCED prompts × 6 models).

  Questions : data/prompts/vqa_questions_ablation_a1.jsonl
  Videos    : data/videos_ablation_a1/<model>/<prompt_id>.mp4
  Output    : data/results/ablation_a1/<model_name>_<prompt_id>.json

Resume-safe: already-written result files are skipped automatically.

Usage:
    python vlm_evaluator_ablation_a1.py
"""

import json
import os
import time
import asyncio
import cv2
import PIL.Image
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))
ENV_PATH     = REPO_ROOT / ".env"
VQA_QUESTIONS= DATA_ROOT / "prompts/vqa_questions_ablation_a1.jsonl"
VIDEO_ROOT   = DATA_ROOT / "videos_ablation_a1"
OUTPUT_DIR   = DATA_ROOT / "results/ablation_a1"

load_dotenv(ENV_PATH)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ── Model directories ─────────────────────────────────────────────────────────
MODEL_DIRS = [
    "fal-ai_veo3.1_fast",
    "fal-ai_kling-video_v2.6_pro_text-to-video",
    "fal-ai_ltx-2_text-to-video",
    "fal-ai_wan_v2.2-a14b_text-to-video",
    "fal-ai_hunyuan-video-v1.5_text-to-video",
    "wan2.1-1.3b",
]

# ── Frame-sampling mode per dimension (identical to main evaluator) ────────────
DIMENSION_MODES = {
    "semantic_adherence":  "holistic",
    "composition_framing": "holistic",
    "lighting_volumetric": "holistic",
    "color_harmony":       "holistic",
    "structural_gestalt":  "holistic",
    "dynamic_degree":      "holistic",
    "semantic_drift":      "holistic",
    "scene_consistency":   "sampled",
    "object_permanence":   "sampled",
    "subject_consistency": "sampled",
    "motion_smoothness":   "micro",
    "temporal_flickering": "micro",
    "inertial_consistency":"micro",
    "physical_mechanics":  "micro",
    "human_fidelity":      "micro",
    "spatial_relationship":"sampled",
}


# ── Frame extraction ───────────────────────────────────────────────────────────
def extract_frames(video_path, mode="sampled", num_frames=16):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    if mode == "sampled":
        indices = [int(i * total / num_frames) for i in range(num_frames)]
    elif mode == "micro":
        indices = list(range(0, min(total, 60), 5))
    else:  # holistic
        indices = [int(i * total / 32) for i in range(min(32, total))]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


# ── Evaluate one dimension ─────────────────────────────────────────────────────
async def evaluate_dimension(model, video_path, questions, dimension, prompt_id, model_name):
    mode = DIMENSION_MODES.get(dimension, "holistic")
    print(f"  [{model_name}][{prompt_id}] {dimension} ({mode})...", flush=True)

    frames = await asyncio.to_thread(extract_frames, video_path, mode)

    prompt = (
        f"You are evaluating a video generated from a prompt. "
        f"Dimension: {dimension}\n\nQuestions to answer:\n"
    )
    for i, q in enumerate(questions):
        prompt += f"{i+1}. {q['question']}\n   Rubric: {q['rubric_description']}\n"
    prompt += (
        "\nAnswer each question with a score (1-5) and a short justification. "
        "Return ONLY a JSON list of objects: [{\"score\": X, \"justification\": \"...\"}]"
    )

    for attempt in range(5):
        try:
            response = await asyncio.to_thread(model.generate_content, [prompt] + frames)
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            return dimension, json.loads(text)
        except Exception as e:
            if "429" in str(e):
                wait = (2 ** attempt) * 10
                print(f"  [{model_name}][{prompt_id}] rate-limit on {dimension}, wait {wait}s", flush=True)
                await asyncio.sleep(wait)
            else:
                print(f"  [{model_name}][{prompt_id}] error on {dimension}: {e}", flush=True)
                await asyncio.sleep(5)
    return dimension, None


# ── Evaluate one video (dims in batches of 2) ─────────────────────────────────
CONCURRENCY = 1
SEM = None


async def process_video(model, video_path, vqa_set, prompt_id, model_name):
    output_file = OUTPUT_DIR / f"{model_name}_{prompt_id}.json"
    if output_file.exists():
        print(f"  Skip (already done): {model_name}/{prompt_id}", flush=True)
        return

    async with SEM:
        print(f"\nProcessing: {model_name} / {prompt_id}", flush=True)
        tasks = [
            evaluate_dimension(model, video_path, questions, dim, prompt_id, model_name)
            for dim, questions in vqa_set.items()
        ]
        results = {}
        for i in range(0, len(tasks), 2):
            batch = await asyncio.gather(*tasks[i:i+2])
            for dim, res in batch:
                results[dim] = res
            await asyncio.sleep(2)

        with open(output_file, "w") as f:
            json.dump({"model": model_name, "prompt_id": prompt_id, "results": results}, f, indent=2)
        print(f"  DONE → {output_file.name}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────
async def main():
    global SEM
    SEM = asyncio.Semaphore(CONCURRENCY)

    model = genai.GenerativeModel("models/gemini-flash-latest")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    questions_map = {}
    with open(VQA_QUESTIONS) as f:
        for line in f:
            d = json.loads(line)
            questions_map[d["prompt_id"]] = d["vqa"]
    print(f"Loaded questions for {len(questions_map)} prompts")

    video_tasks = []
    for model_dir in MODEL_DIRS:
        dir_path = VIDEO_ROOT / model_dir
        if not dir_path.exists():
            print(f"WARNING: directory not found — {dir_path}")
            continue
        for video_file in sorted(dir_path.glob("*.mp4")):
            pid = video_file.stem
            if pid in questions_map:
                video_tasks.append(
                    process_video(model, video_file, questions_map[pid], pid, model_dir)
                )
            else:
                print(f"  No questions for {pid} — skipping")

    print(f"\nTotal tasks: {len(video_tasks)}  (semaphore={CONCURRENCY})\n")
    await asyncio.gather(*video_tasks)
    print("\nAll done!")


if __name__ == "__main__":
    asyncio.run(main())
