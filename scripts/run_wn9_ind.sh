#!/usr/bin/env bash
set -e
python train.py --config configs/wn9_ind.yaml
python build_memory.py --config configs/wn9_ind.yaml --checkpoint outputs/WN9_ind/checkpoints/best.pt
python test.py --config configs/wn9_ind.yaml --checkpoint outputs/WN9_ind/checkpoints/best.pt --memory outputs/WN9_ind/memory/tree_memory.pt
