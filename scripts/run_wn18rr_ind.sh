#!/usr/bin/env bash
set -e
python train.py --config configs/wn18rr_ind.yaml
python build_memory.py --config configs/wn18rr_ind.yaml --checkpoint outputs/WN18RR_ind/checkpoints/best.pt
python test.py --config configs/wn18rr_ind.yaml --checkpoint outputs/WN18RR_ind/checkpoints/best.pt --memory outputs/WN18RR_ind/memory/tree_memory.pt
