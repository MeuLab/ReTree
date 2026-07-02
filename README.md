# ReTree: Reliability-guided Tree Memory for Inductive Multimodal Knowledge Graph Completion

This repository provides a PyTorch implementation template for **ReTree**, a reliability-guided tree memory framework for inductive multimodal knowledge graph completion (IMKGC). The implementation is organized according to the main components of the paper:

1. **Reliability-aware Representation Learning**
2. **Hierarchical Tree Memory with Adaptive Prototypes**
3. **Hierarchical Tree Retrieval and Inductive Reasoning**

The code is designed to make the method structure explicit. Dataset files are not included in this package. Please place the processed datasets under `data/` following the format described in `data/README.md`.

---

## Requirements

The code is based on Python 3.8+ and PyTorch.

```bash
pip install -r requirements.txt
```

Recommended dependencies:

- `torch >= 1.12.0`
- `numpy >= 1.21.0`
- `scikit-learn >= 1.0.0`
- `tqdm`
- `PyYAML`
- `transformers` (optional, for PLM-based text encoding)

---

## Repository Structure

```text
ReTree/
├── configs/                         # Dataset and ablation configurations
│   ├── fb15k237_ind.yaml
│   ├── wn18rr_ind.yaml
│   ├── wn9_ind.yaml
│   └── ablations/
├── data/                            # Put datasets and pretrained features here
├── retree/
│   ├── common/                      # Config loading, metrics, seed, logging
│   ├── data/                        # Dataset schema, collator, negative sampling
│   ├── encoders/                    # Text, visual, and fused entity encoders
│   ├── reliability/                 # EMA branch, calibration, local separation
│   ├── representation/              # Soft targets and representation losses
│   ├── memory/                      # Adaptive prototypes and tree memory index
│   ├── retrieval/                   # Query-local insertion and hierarchical retrieval
│   ├── models/                      # Full ReTree model and scorer
│   ├── evaluation/                  # Filtered ranking and analysis utilities
│   └── runners/                     # Train, build-memory, and evaluate runners
├── scripts/                         # Example scripts
├── train.py                         # Training entry
├── build_memory.py                  # Tree memory construction entry
├── test.py                          # Evaluation entry
└── preprocess.py                    # Dataset format checking and indexing
```

---

## Data Preparation

Create one folder for each dataset under `data/`:

```text
data/
├── FB15K237_ind/
├── WN18RR_ind/
└── WN9_ind/
```

Each dataset folder should contain:

```text
entities.txt              # entity_id<TAB>entity_name
relations.txt             # relation_id<TAB>relation_name
train.tsv                 # head<TAB>relation<TAB>tail
valid.tsv                 # head<TAB>relation<TAB>tail
test.tsv                  # head<TAB>relation<TAB>tail
entity_text.json          # entity_id -> textual description
entity_image.npy          # entity visual feature matrix
image_index.json          # entity_id -> row index in entity_image.npy
```

The loader also supports pre-computed text and image embeddings. See `data/README.md` for details.

---

## Training

```bash
python train.py --config configs/fb15k237_ind.yaml
```

During training, ReTree learns multimodal entity representations and reliability-aware supervision. The main representation objective contains:

- reliability-aware soft target construction;
- perturbation confidence regularization;
- neighborhood consistency loss;
- EMA-based reliability assessment.

---

## Build Tree Memory

After training the representation module, build the hierarchical tree memory:

```bash
python build_memory.py --config configs/fb15k237_ind.yaml --checkpoint outputs/FB15K237_ind/checkpoints/best.pt
```

The tree memory contains relation roots, adaptive prototypes, and entity nodes. The number of prototypes is computed from relation frequency and intra-relation dispersion.

---

## Evaluation

```bash
python test.py --config configs/fb15k237_ind.yaml --checkpoint outputs/FB15K237_ind/checkpoints/best.pt --memory outputs/FB15K237_ind/memory/tree_memory.pt
```

During inference, ReTree applies query-local insertion or temporary prototype construction for unseen entities, then performs prototype-level retrieval, entity-level expansion, and final reranking.

---

## Example Scripts

```bash
bash scripts/run_fb15k237_ind.sh
bash scripts/run_wn18rr_ind.sh
bash scripts/run_wn9_ind.sh
```

Ablation configurations are provided in `configs/ablations/`.

---

## Citation

If this repository is useful for your research, please cite the corresponding paper.

```bibtex
@article{retree2026,
  title   = {Beyond Flat Retrieval: Reliability-guided Tree Memory for Inductive Multimodal Knowledge Graph Completion},
  author  = {Jingchao Wang and others},
  journal = {Under Review},
  year    = {2026}
}
```
