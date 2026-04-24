"""
vlm_evaluator_claude.py  —  A4: Cross-VLM Auditor Agreement ablation

Evaluates the 20 validation prompts × 6 models using Claude instead of
Gemini, with identical VQA questions and dimension-aware frame sampling.
Results are compared against Gemini's scores to compute inter-VLM
Spearman ρ and confirm auditor-independence of the rankings.

Videos   : data/videos/<model_dir>/   (all 6 models, 20 validation prompts)
Questions: data/prompts/vqa_questions_50prompts.jsonl
Output   : data/results/claude_vlm/<model>_<prompt_id>.json

Usage:
  python vlm_evaluator_claude.py

Resume-safe: existing result files are skipped automatically.
"""

import json
import os
import time
import base64
import asyncio
import io
import cv2
import PIL.Image
from pathlib import Path
from dotenv import load_dotenv
import anthropic

# ── Config ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))
ENV_PATH   = REPO_ROOT / ".env"
OUTPUT_DIR = DATA_ROOT / "results/claude_vlm"
CLAUDE_MODEL = "claude-sonnet-4-6"

# A4 ablation: validation set only (20 prompts) — matches Gemini main study
VQA_SOURCES = [
    str(DATA_ROOT / "prompts/vqa_questions_50prompts.jsonl"),  # 50 prompts
]

VIDEO_SOURCES = [
    {
        # All 6 models (5 FAL-API + wan2.1-1.3b) stored as model-named subdirs
        "root": DATA_ROOT / "videos",
        "label": "validation",
    },
]

MODEL_DIRS = [
    "fal-ai_veo3.1_fast",
    "fal-ai_kling-video_v2.6_pro_text-to-video",
    "fal-ai_ltx-2_text-to-video",
    "fal-ai_wan_v2.2-a14b_text-to-video",
    "fal-ai_hunyuan-video-v1.5_text-to-video",
    "wan2.1-1.3b",
]

# ── Frame-sampling mode per dimension (identical to Gemini evaluator) ──────────
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
    "spatial_relationship":"sampled",
    "motion_smoothness":   "micro",
    "temporal_flickering": "micro",
    "inertial_consistency":"micro",
    "physical_mechanics":  "micro",
    "human_fidelity":      "micro",
}

CONCURRENCY = 2   # videos processed in parallel
SEM         = None

load_dotenv(ENV_PATH)
CLIENT = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))


# ── Frame extraction (identical logic to Gemini evaluator) ─────────────────────
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


def pil_to_base64(img: PIL.Image.Image) -> str:
    """Convert PIL image to base64 JPEG string."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


# ── Evaluate one dimension via Claude ──────────────────────────────────────────
def evaluate_dimension_sync(video_path, questions, dimension, prompt_id, model_name):
    mode   = DIMENSION_MODES.get(dimension, "holistic")
    frames = extract_frames(video_path, mode)

    # Build content blocks: images first, then text prompt
    content = []
    for img in frames:
        content.append({
            "type": "image",
            "source": {
                "type":       "base64",
                "media_type": "image/jpeg",
                "data":       pil_to_base64(img),
            },
        })

    q_text = (
        f"You are evaluating a video generated from a prompt. "
        f"Dimension: {dimension}\n\nQuestions to answer:\n"
    )
    for i, q in enumerate(questions):
        q_text += f"{i+1}. {q['question']}\n   Rubric: {q['rubric_description']}\n"
    q_text += (
        "\nAnswer each question with a score (1-5) and a short justification. "
        'Return ONLY a JSON list of objects: [{"score": X, "justification": "..."}]'
    )
    content.append({"type": "text", "text": q_text})

    for attempt in range(5):
        try:
            resp = CLIENT.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": content}],
            )
            text = resp.content[0].text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            return dimension, json.loads(text)
        except anthropic.RateLimitError:
            wait = (2 ** attempt) * 15
            print(f"  [{model_name}][{prompt_id}] rate-limit on {dimension}, wait {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"  [{model_name}][{prompt_id}] error on {dimension}: {e}", flush=True)
            time.sleep(5)
    return dimension, None


async def evaluate_dimension(video_path, questions, dimension, prompt_id, model_name):
    return await asyncio.to_thread(
        evaluate_dimension_sync, video_path, questions, dimension, prompt_id, model_name
    )


# ── Evaluate one video (all dimensions, batches of 2) ─────────────────────────
async def process_video(video_path, vqa_set, prompt_id, model_name):
    output_file = OUTPUT_DIR / f"{model_name}_{prompt_id}.json"
    if output_file.exists():
        print(f"  Skip (done): {model_name}/{prompt_id}", flush=True)
        return

    async with SEM:
        print(f"\nProcessing: {model_name} / {prompt_id}", flush=True)
        tasks = [
            evaluate_dimension(video_path, questions, dim, prompt_id, model_name)
            for dim, questions in vqa_set.items()
        ]
        results = {}
        for i in range(0, len(tasks), 2):
            batch = await asyncio.gather(*tasks[i:i+2])
            for dim, res in batch:
                results[dim] = res
            await asyncio.sleep(3)   # conservative rate-limit buffer

        with open(output_file, "w") as f:
            json.dump({
                "model":     model_name,
                "prompt_id": prompt_id,
                "vlm":       CLAUDE_MODEL,
                "results":   results,
            }, f, indent=2)
        print(f"  DONE → {output_file.name}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────
async def main():
    global SEM
    SEM = asyncio.Semaphore(CONCURRENCY)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all 50 prompt question sets
    questions_map = {}
    for qfile in VQA_SOURCES:
        with open(qfile) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                pid = d.get("prompt_id")
                if pid and d.get("vqa"):
                    questions_map[pid] = d["vqa"]
    print(f"Loaded questions for {len(questions_map)} prompts")

    # Build task list from all video sources
    video_tasks = []
    seen = set()
    for source in VIDEO_SOURCES:
        root = source["root"]
        for model_dir in MODEL_DIRS:
            dir_path = root / model_dir
            if not dir_path.exists():
                continue
            for video_file in sorted(dir_path.glob("*.mp4")):
                pid = video_file.stem
                key = f"{model_dir}_{pid}"
                if key in seen:
                    continue          # don't double-count if somehow in both dirs
                seen.add(key)
                if pid not in questions_map:
                    print(f"  No questions for {pid} — skipping")
                    continue
                video_tasks.append(
                    process_video(video_file, questions_map[pid], pid, model_dir)
                )

    done_already = sum(
        1 for t in video_tasks
        # rough count — actual skip check is inside process_video
    )
    print(f"\nTotal tasks queued: {len(video_tasks)}  (concurrency={CONCURRENCY})\n")
    await asyncio.gather(*video_tasks)
    print("\nAll done!")


if __name__ == "__main__":
    asyncio.run(main())
