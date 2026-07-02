# RoboTwin / DP3 Integration

This folder contains a lightweight patch for using exported semantic point clouds
inside RoboTwin/DP3. It does not vendor RoboTwin or DP3; the files here are
copied into an existing RoboTwin checkout.

## Expected External Layout

Assume the RoboTwin repository is located at:

```text
/path/to/RoboTwin_geo
```

and this release package is located at:

```text
/path/to/semantic_field_release
```

Create a link from RoboTwin to this release package:

```bash
cd /path/to/RoboTwin_geo
mkdir -p include
ln -s /path/to/semantic_field_release include/3d_semantic_train
```

The semantic feature wrapper expects the release package at:

```text
RoboTwin_geo/include/3d_semantic_train
```

## Copy Integration Files

Copy the release integration patch into RoboTwin's DP3 folder:

```bash
cd /path/to/RoboTwin_geo/policy/DP3
cp /path/to/semantic_field_release/integrations/dp3/*.sh ./
cp /path/to/semantic_field_release/integrations/dp3/*.py ./scripts/
cp /path/to/semantic_field_release/integrations/dp3/config/robot_dp3_semantic_pointwise_hybrid.yaml \
  ./3D-Diffusion-Policy/diffusion_policy_3d/config/
cp /path/to/semantic_field_release/integrations/dp3/config/task/demo_task_semantic_pointwise_hybrid.yaml \
  ./3D-Diffusion-Policy/diffusion_policy_3d/config/task/
```

The training script dynamically adds observation keys such as
`semantic_point_cloud_A` and `semantic_point_cloud_B` to the DP3 Hydra config.
For a 128-D semantic embedding and 256 query points, each semantic point cloud
has shape `[256, 131]`, where `131 = 3 xyz channels + 128 embedding channels`.

## Train DP3 with Semantic Point Clouds

Single-object task example:

```bash
cd /path/to/RoboTwin_geo/policy/DP3

bash train_semantic_pointwise_hybrid.sh \
  hanging_mug \
  demo_clean_3d \
  50 \
  0 \
  0 \
  /path/to/object_A_field/best.pt \
  none \
  cuda:0 \
  "{A}" \
  256 \
  128
```

Argument order:

```text
task_name
task_config
expert_data_num
seed
gpu_id
semantic_ckpt_A
semantic_ckpt_B
semantic_device
object_placeholders
semantic_point_num
semantic_feat_dim
```

For two-object tasks, pass both checkpoints and placeholders:

```bash
bash train_semantic_pointwise_hybrid.sh \
  task_name demo_clean_3d 50 0 0 \
  /path/to/object_A_field/best.pt \
  /path/to/object_B_field/best.pt \
  cuda:0 \
  "{A},{B}" \
  256 \
  128
```

## What the Preprocessing Does

The preprocessing script reads each demonstration episode, extracts target object
point clouds, queries the frozen semantic field, and writes additional zarr
observations:

```text
semantic_point_cloud_A
semantic_point_cloud_B
```

Each semantic point cloud has shape:

```text
[semantic_point_num, 3 + semantic_feat_dim]
```

DP3 then treats each semantic point cloud as an additional point-cloud modality.
The diffusion policy objective and action representation are unchanged.

## Real-Robot Deployment

For real-robot experiments, the same interface is used online:

1. segment object masks from RGB-D observations,
2. back-project masks to object point clouds,
3. call `compute_semantic_pointwise_cloud(...)`,
4. insert the resulting semantic point clouds into the DP3 observation dict.

The release includes `semantic_feature_utils.py` as the reference implementation
of steps 3 and 4.
