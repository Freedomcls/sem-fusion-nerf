#!/bin/bash

python scripts/inference3d-indoor.py \
  --exp_dir=out/indoor \
  --dataset_type=replica_seg_to_3dface \
  --pigan_curriculum_type=replica \
  --checkpoint_path=ckpts/best_model.pt \
  --data_path=data/semantic-masks/ \
  --test_output_size=256 \
  --pigan_infer_ray_step=72 \
  --latent_mask=99 \
  --inject_code_seed=890234
