# Object-Centric Continuous Semantic Field

This is a cleaned release package for training and using the object-centric
continuous semantic field described in our CoRL submission. The code keeps only
the components needed for:

- training a PartNext-supervised semantic field,
- querying the trained field to export `xyz + semantic embedding` point clouds,
- visualizing learned semantic embeddings and support/query points,
- integrating exported semantic point clouds with RoboTwin/DP3.

The original research workspace contained many exploratory branches. They are
intentionally not included here.

## Repository Layout

```text
configs/                         # canonical part alias configs
models/                          # semantic-field model and Utonia feature extractor
my_datasets/                     # PartNext semantic-field dataset loader
myutils/                         # small training/runtime utilities
train/                           # semantic-field training entry point
tools/                           # qualitative visualization tools
scripts/                         # release-friendly wrappers and export script
integrations/dp3/                # minimal RoboTwin/DP3 integration notes and scripts
```

## Environment

Install PyTorch for your CUDA version first, then install the remaining Python
dependencies:

```bash
pip install -r requirements.txt
```

The code expects the Utonia package/checkpoint to be available. You can either
install Utonia in the Python environment or place a checkout at:

```text
include/Utonia
```

The default Utonia checkpoint path is:

```text
~/.cache/utonia/ckpt/utonia.pth
```

You can override it with `--utonia-checkpoint` or the `UTONIA_CHECKPOINT`
environment variable used by the shell wrappers.

## Data

Semantic field training uses PartNext object models with part annotations. The
expected dataset root is the PartNext mesh directory, e.g.

```text
/path/to/PartNext_mesh/
  Hammer/
  Mug/
  Spoon/
  ...
```

Part aliases are controlled by JSON files in `configs/`. For example,
`configs/hammer.json` maps raw PartNext labels to the category-specific part
label space used by the field.

## Train a Semantic Field

Example for a hammer field:

```bash
DATASET_ROOT=/path/to/PartNext_mesh \
UTONIA_CHECKPOINT=~/.cache/utonia/ckpt/utonia.pth \
CATEGORIES=Hammer \
ALIAS_CONFIG=configs/hammer.json \
RUN_NAME=hammer_semantic_field \
WANDB_MODE=disabled \
bash scripts/train_semantic_field.sh
```

The wrapper calls:

```bash
python -m train.train_utonia_universal_field --train-mode semantic ...
```

Important outputs:

- `best.pt`: best validation checkpoint,
- `last.pt`: latest checkpoint,
- `canonical_labels.json`: part label names used by the checkpoint.

## Export Semantic Point Clouds

After training, query the frozen field on an object point cloud:

```bash
python scripts/export_semantic_point_cloud.py \
  --checkpoint outputs/semantic_field/hammer_semantic_field/best.pt \
  --input-point-cloud /path/to/object_point_cloud.npy \
  --output-npz outputs/hammer_semantic_point_cloud.npz \
  --num-query-points 256 \
  --device cuda
```

The output `.npz` contains:

- `semantic_point_cloud`: shape `[N, 3 + D]`, where the first three channels are
  world/object coordinates and the remaining channels are semantic embeddings,
- `query_xyz`,
- `sem_embeddings`,
- `sem_logits`,
- `sem_probabilities`,
- `pred_labels`,
- `confidence`,
- `label_names`.

This is the representation consumed by downstream point-cloud policies.

## Visualization

Visualize predictions and semantic embedding PCA on PartNext samples:

```bash
CHECKPOINT=outputs/semantic_field/hammer_semantic_field/best.pt \
DATASET_ROOT=/path/to/PartNext_mesh \
bash scripts/visualize_semantic_field.sh
```

Visualize support points and labeled query points used during training:

```bash
DATASET_ROOT=/path/to/PartNext_mesh \
CATEGORIES=Hammer \
ALIAS_CONFIG=configs/hammer.json \
bash scripts/visualize_support_query_points.sh
```

The support/query visualizer writes an interactive HTML file by default.

## RoboTwin / DP3 Integration

The semantic field is used as a frozen object-level representation module. At
each policy step:

1. obtain an object point cloud from RoboTwin or real RGB-D observations,
2. use it as the support condition for the semantic field,
3. resample object query locations,
4. export `semantic_point_cloud_A`, `semantic_point_cloud_B`, ... as
   `xyz + semantic embedding`,
5. provide those arrays as additional point-cloud modalities to DP3.

See `integrations/dp3/README.md` for the concrete file-copy steps, Hydra config patch, preprocessing command, and training command used with RoboTwin/DP3.

## Notes

- This release focuses on the semantic-field branch. Exploratory occupancy,
  NDF, and geometric-field branches were intentionally removed.
- Checkpoints and datasets are not included.
- `open3d` is only needed for the optional Open3D visualization backend.
