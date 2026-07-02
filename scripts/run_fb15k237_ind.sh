#!/usr/bin/env bash
set -e
python train.py --config configs/fb15k237_ind.yaml
python build_memory.py --config configs/fb15k237_ind.yaml --checkpoint outputs/FB15K237_ind/checkpoints/best.pt
python test.py --config configs/fb15k237_ind.yaml --checkpoint outputs/FB15K237_ind/checkpoints/best.pt --memory outputs/FB15K237_ind/memory/tree_memory.pt
