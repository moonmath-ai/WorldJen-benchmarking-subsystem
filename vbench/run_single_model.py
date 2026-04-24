"""Single-model VBench runner. Usage: python run_single_model.py <model_key> [gpu_id]

Set WORLDJEN_DATA_ROOT and VBENCH_DIR env vars before running (see README).
"""
import os, sys, json, logging
from pathlib import Path

GPU_ID = sys.argv[2] if len(sys.argv) > 2 else '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

_VBENCH_DIR = os.environ.get("VBENCH_DIR", "")
if _VBENCH_DIR:
    sys.path.insert(0, _VBENCH_DIR)

import torch
from vbench import VBench

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

_DATA_ROOT  = Path(os.environ.get("WORLDJEN_DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
VIDEOS_BASE = str(_DATA_ROOT / "videos")
VBENCH_DIR  = _VBENCH_DIR or str(Path(__file__).resolve().parents[1] / "vbench_lib")
EVAL_DIR    = os.path.join(VBENCH_DIR, 'evaluation_results')
FULL_INFO   = os.path.join(VBENCH_DIR, 'vbench', 'VBench_full_info.json')

VBENCH_DIMS = [
    'subject_consistency', 'background_consistency', 'motion_smoothness',
    'dynamic_degree', 'aesthetic_quality', 'imaging_quality',
]

model_key = sys.argv[1]
device = torch.device('cuda')
log.info(f'Model: {model_key}  GPU: {GPU_ID}')

videos_path = os.path.join(VIDEOS_BASE, model_key)

for dim in VBENCH_DIMS:
    output_path = os.path.join(EVAL_DIR, model_key, dim)
    os.makedirs(output_path, exist_ok=True)
    name = f'{model_key}_{dim}'
    results_file = os.path.join(output_path, f'{name}_eval_results.json')

    if os.path.exists(results_file):
        log.info(f'  [CACHED] {dim}')
        continue

    log.info(f'  [RUN] {dim}')
    my_VBench = VBench(device, FULL_INFO, output_path)
    my_VBench.evaluate(
        videos_path=videos_path,
        name=name,
        dimension_list=[dim],
        mode='custom_input',
    )
    log.info(f'  [DONE] {dim}')

log.info(f'All dims complete for {model_key}')
