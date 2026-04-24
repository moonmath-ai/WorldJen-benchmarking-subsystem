"""
VBench + VBench-2.0 evaluation runner for WorldJen paper.
Runs custom_input mode on existing 300 videos (6 models × 50 prompts).
Outputs: data/vbench/vbench_summary.json  (or WORLDJEN_DATA_ROOT/vbench/vbench_summary.json)
"""

import os
import sys
import json
import numpy as np
import logging
from datetime import datetime, timezone
from pathlib import Path

VBENCH_DIR  = os.environ.get('VBENCH_DIR', str(Path.home() / 'VBench'))
sys.path.insert(0, VBENCH_DIR)

import torch
from scipy.stats import spearmanr
import choix

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("WORLDJEN_DATA_ROOT", REPO_ROOT / "data"))
VIDEOS_BASE = str(DATA_ROOT / 'videos')
OUTPUT_DIR  = str(DATA_ROOT / 'vbench')
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'vbench_summary.json')
VBENCH_EVAL_RESULTS_DIR = os.path.join(VBENCH_DIR, 'evaluation_results')

VBENCH_DIMS = [
    'subject_consistency',
    'background_consistency',
    'motion_smoothness',
    'dynamic_degree',
    'aesthetic_quality',
    'imaging_quality',
]

# VBench quality score normalization (from scripts/constant.py)
NORMALIZE_DIC = {
    'subject consistency':     {'Min': 0.1462, 'Max': 1.0},
    'background consistency':  {'Min': 0.2615, 'Max': 1.0},
    'motion smoothness':       {'Min': 0.706,  'Max': 0.9975},
    'dynamic degree':          {'Min': 0.0,    'Max': 1.0},
    'aesthetic quality':       {'Min': 0.0,    'Max': 1.0},
    'imaging quality':         {'Min': 0.0,    'Max': 1.0},
}
DIM_WEIGHT = {
    'subject consistency':    1,
    'background consistency': 1,
    'motion smoothness':      1,
    'dynamic degree':         0.5,
    'aesthetic quality':      1,
    'imaging quality':        1,
}

MODEL_DISPLAY = {
    'fal-ai_veo3.1_fast':                          'Veo 3.1 Fast',
    'fal-ai_kling-video_v2.6_pro_text-to-video':   'Kling v2.6 Pro',
    'fal-ai_wan_v2.2-a14b_text-to-video':          'Wan v2.2 A14B',
    'fal-ai_ltx-2_text-to-video':                  'LTX-2',
    'fal-ai_hunyuan-video-v1.5_text-to-video':     'HunyuanVideo v1.5',
    'wan2.1-1.3b':                                 'Wan 2.1 1.3B',
}

WORLDJEN_BT = {
    'fal-ai_veo3.1_fast':                        1637,
    'fal-ai_kling-video_v2.6_pro_text-to-video': 1615,
    'fal-ai_wan_v2.2-a14b_text-to-video':        1508,
    'fal-ai_ltx-2_text-to-video':               1468,
    'fal-ai_hunyuan-video-v1.5_text-to-video':  1449,
    'wan2.1-1.3b':                               1324,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VBENCH_EVAL_RESULTS_DIR, exist_ok=True)


# ── VBench evaluation ─────────────────────────────────────────────────────────

def run_vbench_dimension(model_key, dimension, device):
    """Run one VBench dimension on one model's videos. Returns (mean_score, video_results_list)."""
    from vbench import VBench

    videos_path = os.path.join(VIDEOS_BASE, model_key)
    output_path = os.path.join(VBENCH_EVAL_RESULTS_DIR, model_key, dimension)
    os.makedirs(output_path, exist_ok=True)

    full_info_dir = os.path.join(VBENCH_DIR, 'vbench', 'VBench_full_info.json')
    my_VBench = VBench(device, full_info_dir, output_path)

    name = f'{model_key}_{dimension}'
    results_file = os.path.join(output_path, f'{name}_eval_results.json')

    # Resume if already computed
    if os.path.exists(results_file):
        log.info(f'  [CACHED] {model_key} / {dimension}')
        with open(results_file) as f:
            saved = json.load(f)
        mean_score, video_results = saved[dimension]
        return mean_score, video_results

    log.info(f'  [RUN] {model_key} / {dimension}')
    my_VBench.evaluate(
        videos_path=videos_path,
        name=name,
        dimension_list=[dimension],
        mode='custom_input',
    )

    with open(results_file) as f:
        saved = json.load(f)
    mean_score, video_results = saved[dimension]
    return mean_score, video_results


def run_all_vbench(device):
    """Run all 6 VBench dims for all 6 models. Returns nested dict."""
    model_keys = sorted([
        d for d in os.listdir(VIDEOS_BASE)
        if os.path.isdir(os.path.join(VIDEOS_BASE, d))
    ])

    results = {}  # {model_key: {dim: {'mean': float, 'per_video': {filename: score}}}}

    for model_key in model_keys:
        results[model_key] = {}
        log.info(f'=== Model: {model_key} ===')
        for dim in VBENCH_DIMS:
            mean_score, video_results = run_vbench_dimension(model_key, dim, device)
            # Build filename → score mapping
            per_video = {}
            for vr in video_results:
                vpath = vr['video_path']
                score = vr['video_results']
                fname = os.path.basename(vpath)
                per_video[fname] = float(score)
            results[model_key][dim] = {
                'mean': float(mean_score),
                'per_video': per_video,
            }
            log.info(f'    {dim}: {mean_score:.4f}')

    return results


# ── Quality score (normalized, VBench formula) ────────────────────────────────

def compute_quality_score(dim_means):
    """Compute normalized quality score for our 6 dims (no temporal_flickering)."""
    total_weight = 0
    total_score  = 0
    for dim in VBENCH_DIMS:
        key = dim.replace('_', ' ')
        if key not in NORMALIZE_DIC:
            continue
        raw = dim_means[dim]
        mn  = NORMALIZE_DIC[key]['Min']
        mx  = NORMALIZE_DIC[key]['Max']
        norm = (raw - mn) / (mx - mn)
        norm = max(0.0, min(1.0, norm))
        w   = DIM_WEIGHT[key]
        total_score  += norm * w
        total_weight += w
    return total_score / total_weight if total_weight > 0 else 0.0


# ── Bradley-Terry BT rating ─────────────────────────────────────────────────────────

def compute_bt_rating(vbench_results, model_keys):
    """
    Build pairwise win data from per-video scores across all 6 dims.
    All 6 models share the same 50 video filenames (WorldJen prompts).
    Returns dict of {model_key: bt_score}.
    """
    n = len(model_keys)
    idx = {m: i for i, m in enumerate(model_keys)}

    # Collect pairwise wins: list of (winner_idx, loser_idx)
    pairwise_data = []

    # Get common filenames (intersection across all models and dims)
    all_filenames = None
    for mk in model_keys:
        for dim in VBENCH_DIMS:
            fnames = set(vbench_results[mk][dim]['per_video'].keys())
            all_filenames = fnames if all_filenames is None else all_filenames & fnames

    log.info(f'BT: {len(all_filenames)} common videos across all models')

    for fname in sorted(all_filenames):
        for dim in VBENCH_DIMS:
            scores = {mk: vbench_results[mk][dim]['per_video'].get(fname, None)
                      for mk in model_keys}
            # All pairs
            for i, ma in enumerate(model_keys):
                for j, mb in enumerate(model_keys):
                    if j <= i:
                        continue
                    sa = scores[ma]
                    sb = scores[mb]
                    if sa is None or sb is None:
                        continue
                    if sa > sb:
                        pairwise_data.append((idx[ma], idx[mb]))
                    elif sb > sa:
                        pairwise_data.append((idx[mb], idx[ma]))
                    # ties: skip

    log.info(f'BT: {len(pairwise_data)} pairwise comparisons')

    if not pairwise_data:
        return {mk: 1000 for mk in model_keys}

    # Fit BT model
    params = choix.ilsr_pairwise(n, pairwise_data, alpha=0.01)

    # Convert to BT rating scale: anchor weakest at 1000
    min_param = params.min()
    bt_scale = 400 / np.log(10)  # standard BT rating: 400 points = 10x stronger
    bt_scores = {mk: round(1000 + (params[idx[mk]] - min_param) * bt_scale, 1)
                  for mk in model_keys}

    return bt_scores


# ── Spearman ρ ────────────────────────────────────────────────────────────────

def compute_spearman(vbench_bt, worldjen_bt, model_keys):
    """Compute Spearman ρ between VBench BT rating rank and WorldJen BT rank."""
    vb_scores = [vbench_bt[mk] for mk in model_keys]
    wj_scores = [worldjen_bt[mk] for mk in model_keys]
    rho, pval = spearmanr(vb_scores, wj_scores)
    return float(rho), float(pval)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')

    # ── Step 1: Run VBench ────────────────────────────────────────────────────
    log.info('=== Running VBench (6 dims, custom_input) ===')
    vbench_results = run_all_vbench(device)
    model_keys = sorted(vbench_results.keys())

    # ── Step 2: Compute mean scores & quality_score ───────────────────────────
    model_vbench_scores = {}
    for mk in model_keys:
        dim_means = {dim: vbench_results[mk][dim]['mean'] for dim in VBENCH_DIMS}
        qs = compute_quality_score(dim_means)
        model_vbench_scores[mk] = {**dim_means, 'quality_score': qs}

    # ── Step 3: VBench BT rating (pairwise BT MLE over per-video scores) ─────────
    log.info('=== Computing VBench BT rating ===')
    vbench_bt = compute_bt_rating(vbench_results, model_keys)
    log.info('VBench BT rating: ' + str({MODEL_DISPLAY.get(k, k): v for k, v in vbench_bt.items()}))

    # ── Step 4: Rankings ─────────────────────────────────────────────────────
    vbench_bt_rank  = sorted(model_keys, key=lambda k: vbench_bt[k], reverse=True)
    worldjen_bt_rank = sorted(WORLDJEN_BT.keys(), key=lambda k: WORLDJEN_BT[k], reverse=True)

    # ── Step 5: Spearman ρ ───────────────────────────────────────────────────
    # Only include models present in both rankings
    common = [mk for mk in worldjen_bt_rank if mk in vbench_bt]
    rho, pval = compute_spearman(vbench_bt, WORLDJEN_BT, common)
    log.info(f'Spearman ρ = {rho:.4f}, p = {pval:.4f}')

    # ── Step 6: Build output JSON ─────────────────────────────────────────────
    out = {
        'metadata': {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'videos_path': VIDEOS_BASE,
            'n_videos_per_model': 50,
            'evaluation_mode': 'custom_input (WorldJen prompts, not VBench standard suite)',
            'vbench_dimensions': VBENCH_DIMS,
            'vbench2_dimensions': ['Human_Anatomy'],
            'note_vbench2': 'Human_Anatomy scores added separately after vbench2 env run',
        },
        'models': {},
        'rankings': {
            'vbench_bt_rank': vbench_bt_rank,
            'vbench2_human_anatomy_rank': [],  # filled after vbench2 run
        },
        'cross_benchmark': {
            'worldjen_bt_rank': worldjen_bt_rank,
            'vbench_bt_rank': vbench_bt_rank,
            'spearman_rho': round(rho, 4),
            'p_value': round(pval, 4),
        },
        'per_video_scores': {},
    }

    for mk in model_keys:
        dim_scores = model_vbench_scores[mk]
        vbench_entry = {dim: round(dim_scores[dim], 6) for dim in VBENCH_DIMS}
        vbench_entry['quality_score'] = round(dim_scores['quality_score'], 6)

        out['models'][mk] = {
            'display_name': MODEL_DISPLAY.get(mk, mk),
            'vbench': vbench_entry,
            'vbench2': {
                'Human_Anatomy': None,  # placeholder — filled after vbench2 run
            },
            'vbench_bt': vbench_bt[mk],
        }

        out['per_video_scores'][mk] = {}
        for dim in VBENCH_DIMS:
            # Store as sorted list of [filename, score] for readability
            pv = vbench_results[mk][dim]['per_video']
            out['per_video_scores'][mk][dim] = {
                fname: round(score, 6) for fname, score in sorted(pv.items())
            }

    # ── Step 7: Write JSON ────────────────────────────────────────────────────
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    log.info(f'Written: {OUTPUT_JSON}')

    # Summary table
    log.info('\n=== VBench Summary ===')
    header = f"{'Model':<45} {'SubjC':>6} {'BgC':>6} {'MotSm':>6} {'DynD':>6} {'Aes':>6} {'ImgQ':>6} {'QScore':>8} {'BT rating':>6}"
    log.info(header)
    for mk in vbench_bt_rank:
        sc = out['models'][mk]['vbench']
        log.info(
            f"{MODEL_DISPLAY.get(mk, mk):<45} "
            f"{sc['subject_consistency']:>6.3f} "
            f"{sc['background_consistency']:>6.3f} "
            f"{sc['motion_smoothness']:>6.3f} "
            f"{sc['dynamic_degree']:>6.3f} "
            f"{sc['aesthetic_quality']:>6.3f} "
            f"{sc['imaging_quality']:>6.3f} "
            f"{sc['quality_score']:>8.4f} "
            f"{out['models'][mk]['vbench_bt']:>6.0f}"
        )
    log.info(f"\nSpearman ρ (VBench BT rating vs WorldJen BT) = {rho:.4f}  p = {pval:.4f}")
    log.info(f'\nOutput: {OUTPUT_JSON}')


if __name__ == '__main__':
    main()
