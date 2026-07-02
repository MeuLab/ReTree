# Data Folder

Place the datasets in this folder. The repository does not include datasets or pretrained visual features.

Expected folder layout:

```text
FB15K237_ind/
├── entities.txt
├── relations.txt
├── train.tsv
├── valid.tsv
├── test.tsv
├── entity_text.json
├── entity_image.npy
└── image_index.json
```

The same structure is used for `WN18RR_ind` and `WN9_ind`.

## File Format

`entities.txt`

```text
entity_id<TAB>entity_name
```

`relations.txt`

```text
relation_id<TAB>relation_name
```

`train.tsv`, `valid.tsv`, `test.tsv`

```text
head_entity_id<TAB>relation_id<TAB>tail_entity_id
```

`entity_text.json`

```json
{
  "entity_id": "textual description"
}
```

`image_index.json`

```json
{
  "entity_id": 0
}
```

`entity_image.npy` is a dense matrix with shape `[num_visual_entities, visual_dim]`.
