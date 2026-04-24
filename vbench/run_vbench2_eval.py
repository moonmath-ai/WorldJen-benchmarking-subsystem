"""
VBench-2.0 Human_Anatomy evaluation runner for WorldJen paper.
Runs custom_input mode on existing 300 videos (6 models × 50 prompts).
Merges results into data/vbench/vbench_summary.json  (or WORLDJEN_DATA_ROOT/vbench/vbench_summary.json)
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# Must be run from VBench-2.0 directory so config relative paths work
VBENCH2_DIR = os.environ.get('VBENCH2_DIR', str(Path.home() / 'VBench/VBench-2.0'))
os.chdir(VBENCH2_DIR)

# Add paths needed for VBench-2.0 and YOLO-World
sys.path.insert(0, VBENCH2_DIR)
sys.path.insert(0, os.path.join(VBENCH2_DIR, 'vbench2', 'third_party', 'YOLO-World'))
VBENCH_DIR  = os.environ.get('VBENCH_DIR', str(Path.home() / 'VBench'))
sys.path.insert(0, VBENCH_DIR)

import torch
from vbench2 import VBench2

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))
VIDEOS_BASE = str(DATA_ROOT / 'videos')
OUTPUT_DIR  = str(DATA_ROOT / 'vbench')
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'vbench_summary.json')
VBENCH2_EVAL_RESULTS_DIR = os.path.join(OUTPUT_DIR, 'vbench2_evaluation_results')

MODEL_DISPLAY = {
    'fal-ai_veo3.1_fast':                          'Veo 3.1 Fast',
    'fal-ai_kling-video_v2.6_pro_text-to-video':   'Kling v2.6 Pro',
    'fal-ai_wan_v2.2-a14b_text-to-video':          'Wan v2.2 A14B',
    'fal-ai_ltx-2_text-to-video':                  'LTX-2',
    'fal-ai_hunyuan-video-v1.5_text-to-video':     'HunyuanVideo v1.5',
    'wan2.1-1.3b':                                 'Wan 2.1 1.3B',
}

os.makedirs(VBENCH2_EVAL_RESULTS_DIR, exist_ok=True)


def run_vbench2_human_anatomy(model_key, device):
    """Run VBench-2.0 Human_Anatomy on one model's videos. Returns (mean_score, video_results)."""
    videos_path = os.path.join(VIDEOS_BASE, model_key)
    output_path = os.path.join(VBENCH2_EVAL_RESULTS_DIR, model_key)
    os.makedirs(output_path, exist_ok=True)

    full_info_dir = os.path.join(VBENCH2_DIR, 'vbench2', 'VBench2_full_info.json')
    my_VBench2 = VBench2(device, full_info_dir, output_path)

    name = f'{model_key}_Human_Anatomy'
    results_file = os.path.join(output_path, f'{name}_eval_results.json')

    # Resume if already computed
    if os.path.exists(results_file):
        log.info(f'  [CACHED] {model_key} / Human_Anatomy')
        with open(results_file) as f:
            saved = json.load(f)
        mean_score, video_results = saved['Human_Anatomy']
        return float(mean_score), video_results

    log.info(f'  [RUN] {model_key} / Human_Anatomy')
    my_VBench2.evaluate(
        videos_path=videos_path,
        name=name,
        dimension_list=['Human_Anatomy'],
        mode='custom_input',
    )

    with open(results_file) as f:
        saved = json.load(f)
    mean_score, video_results = saved['Human_Anatomy']
    return float(mean_score), video_results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')

    model_keys = sorted([
        d for d in os.listdir(VIDEOS_BASE)
        if os.path.isdir(os.path.join(VIDEOS_BASE, d))
    ])

    # ── Run Human_Anatomy ────────────────────────────────────────────────────
    log.info('=== Running VBench-2.0 Human_Anatomy (custom_input) ===')
    human_anatomy_results = {}

    for model_key in model_keys:
        log.info(f'=== Model: {model_key} ===')
        mean_score, video_results = run_vbench2_human_anatomy(model_key, device)
        per_video = {}
        for vr in video_results:
            vpath = vr['video_path']
            score = vr['video_results']
            fname = os.path.basename(vpath)
            per_video[fname] = float(score)
        human_anatomy_results[model_key] = {
            'mean': mean_score,
            'per_video': per_video,
        }
        log.info(f'  Human_Anatomy mean: {mean_score:.4f}')

    # ── Rank by Human_Anatomy ────────────────────────────────────────────────
    ha_rank = sorted(model_keys, key=lambda k: human_anatomy_results[k]['mean'], reverse=True)

    # ── Merge into existing vbench_summary.json ──────────────────────────────
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON) as f:
            summary = json.load(f)
    else:
        log.warning(f'{OUTPUT_JSON} not found — creating new file. Run run_vbench_eval.py first!')
        summary = {'models': {}, 'rankings': {}, 'per_video_scores': {}}

    # Update models
    for mk in model_keys:
        if mk not in summary['models']:
            summary['models'][mk] = {
                'display_name': MODEL_DISPLAY.get(mk, mk),
                'vbench': {},
                'vbench2': {},
                'vbench_bt': None,
            }
        summary['models'][mk]['vbench2']['Human_Anatomy'] = round(
            human_anatomy_results[mk]['mean'], 6
        )

        # Per-video scores
        if mk not in summary.get('per_video_scores', {}):
            summary.setdefault('per_video_scores', {})[mk] = {}
        summary['per_video_scores'][mk]['Human_Anatomy'] = {
            fname: round(score, 6)
            for fname, score in sorted(human_anatomy_results[mk]['per_video'].items())
        }

    # Update rankings
    summary['rankings']['vbench2_human_anatomy_rank'] = ha_rank

    # Update metadata
    summary.setdefault('metadata', {})['vbench2_generated_at'] = datetime.now(timezone.utc).isoformat()

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info(f'Updated: {OUTPUT_JSON}')

    # Summary
    log.info('\n=== VBench-2.0 Human_Anatomy Scores ===')
    for mk in ha_rank:
        score = human_anatomy_results[mk]['mean']
        log.info(f'  {MODEL_DISPLAY.get(mk, mk):<40} {score:.4f}')
    log.info(f'\nHuman_Anatomy rank: {[MODEL_DISPLAY.get(k, k) for k in ha_rank]}')


if __name__ == '__main__':
    main()
