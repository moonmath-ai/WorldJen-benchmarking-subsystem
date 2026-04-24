"""Single-model VBench-2.0 Human_Anatomy runner. Usage: python run_single_model_vbench2.py <model_key> [gpu_id]

Set WORLDJEN_DATA_ROOT and VBENCH_DIR env vars before running (see README).
"""
import os, sys, json, logging
from pathlib import Path

GPU_ID = sys.argv[2] if len(sys.argv) > 2 else '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

_VBENCH_DIR = os.environ.get("VBENCH_DIR", "")
VBENCH2_DIR = os.path.join(_VBENCH_DIR, 'VBench-2.0') if _VBENCH_DIR else str(
    Path(__file__).resolve().parents[1] / "vbench_lib" / "VBench-2.0")
os.chdir(VBENCH2_DIR)
sys.path.insert(0, VBENCH2_DIR)
sys.path.insert(0, os.path.join(VBENCH2_DIR, 'vbench2', 'third_party', 'YOLO-World'))
if _VBENCH_DIR:
    sys.path.insert(0, _VBENCH_DIR)

import torch
from vbench2 import VBench2

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

_DATA_ROOT  = Path(os.environ.get("WORLDJEN_DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
VIDEOS_BASE = str(_DATA_ROOT / "videos")
OUTPUT_DIR  = str(_DATA_ROOT / "vbench" / "vbench2_evaluation_results")
FULL_INFO   = os.path.join(VBENCH2_DIR, 'vbench2', 'VBench2_full_info.json')

model_key = sys.argv[1]
device = torch.device('cuda')
log.info(f'VBench-2.0 Model: {model_key}  GPU: {GPU_ID}')

output_path = os.path.join(OUTPUT_DIR, model_key)
os.makedirs(output_path, exist_ok=True)

name = f'{model_key}_Human_Anatomy'
results_file = os.path.join(output_path, f'{name}_eval_results.json')

if os.path.exists(results_file):
    log.info(f'  [CACHED] Human_Anatomy')
    with open(results_file) as f:
        r = json.load(f)
    mean, _ = r['Human_Anatomy']
    log.info(f'  Cached mean: {mean:.4f}')
    sys.exit(0)

log.info(f'  [RUN] Human_Anatomy')
vb2 = VBench2(device, FULL_INFO, output_path)
vb2.evaluate(
    videos_path=os.path.join(VIDEOS_BASE, model_key),
    name=name,
    dimension_list=['Human_Anatomy'],
    mode='custom_input',
)

with open(results_file) as f:
    r = json.load(f)
mean, per_vid = r['Human_Anatomy']
log.info(f'  [DONE] Human_Anatomy mean: {mean:.4f} ({len(per_vid)} videos)')
