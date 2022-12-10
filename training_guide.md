①nerf_pl + triplane
run：python train.py --dataset_name replica --root_dir room_0/Sequence_1/ --N_importance 64 --img_wh 320 240 --num_epochs 60 --batch_size 1024  --lr 1e-5 --lr_scheduler steplr --decay_step 30 50 --decay_gamma 0.5 --exp_name debug_replica-lr-5  --chunk 40000 --mode eg3d --num_gpus 4

②sem2nerf
1. catmask:
python -m torch.distributed.launch --nproc_per_node=2  scripts/train3d.py --exp_dir=out/cat --dataset_type=catmask_seg_to_3dface  --pigan_curriculum_type=CatMask --train_paths_conf=data/CatMask/train_paths.txt  --test_paths_conf=data/CatMask/val_paths.txt --label_nc=8  --input_nc=10  --workers=1  --batch_size=2  --dis_lambda=0 --w_norm_lambda=0 --train_rand_pose_prob=0.5  --use_contour  --use_merged_labels  --patch_train --ray_min_scale=0.08 --start_from_latent_avg

runtime error: address already in use
--master_port 29501
2. chair
①add semantic information：
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=4    --master_port 29501  scripts/train3d.py --exp_dir=out/nerf-chair  --dataset_type=chair_seg_to_3dface --pigan_curriculum_type=chair  --train_paths_conf=data/chair/train_paths.txt --test_paths_conf=data/chair/test_paths.txt --label_nc=41 --input_nc=41 --workers=0   --batch_size=1 --dis_lambda=0 --w_norm_lambda=0  --train_rand_pose_prob=0.5 --patch_train --ray_min_scale=0.08  --train_decoder True >1121chair.log 2>&1 &
②without swinT：

3. replica
①add semantic information
nohup python -m torch.distributed.launch --nproc_per_node=2  scripts/train3d.py --exp_dir=out/onlynerf-indoor --dataset_type=replica_seg_to_3dface --pigan_curriculum_type=replica  --train_paths_conf=data/Sequence_1/train_paths.txt --test_paths_conf=data/Sequence_1/test_paths.txt --label_nc=99 --input_nc=99 --workers=1   --batch_size=1 --dis_lambda=0 --w_norm_lambda=0  --train_rand_pose_prob=0.5 --patch_train  --l1_lambda=1.0  --ray_min_scale=0.08  --train_decoder True   >1122replica.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7  
CUDA_VISIBLE_DEVICES=4,5,6,7  nohup   python -m torch.distributed.launch --nproc_per_node=4  --master_port 29502  scripts/train3d.py --exp_dir=out/rand-replica  --dataset_type=replica_seg_to_3dface --pigan_curriculum_type=replica  --train_paths_conf=data/Sequence_1/train_paths.txt --test_paths_conf=data/Sequence_1/test_paths.txt --label_nc=99 --input_nc=99 --workers=1   --batch_size=1 --dis_lambda=0 --w_norm_lambda=0  --train_rand_pose_prob=0.5 --patch_train --ray_min_scale=0.08  --train_decoder True    --l1_lambda=1.0   >1109randreplica.log 2>&1 &

inference
cat：
CUDA_VISIBLE_DEVICES=6,7 python scripts/inference3d.py  --exp_dir=depth/cat  --dataset_type=catmask_seg_to_3dface  --pigan_curriculum_type=CatMask  --checkpoint_path=pretrained_models/sem2nerf_catmask_pretrained.pt  --data_path=data/CatMask/mask_samples  --test_output_size=512  --pigan_infer_ray_step=72  --use_merged_labels  --use_original_pose  --latent_mask=7,8  --inject_code_seed=390234  --render_videos

replica：
CUDA_VISIBLE_DEVICES=6,7 python scripts/inference3d-indoor.py --exp_dir=out/indoor  --dataset_type=replica_seg_to_3dface --pigan_curriculum_type=replica --checkpoint_path=out/semantic-indoor-box50/checkpoints/iteration_320000.pt --data_path=data/Sequence_1/semantic_samples/ --test_output_size=256 --pigan_infer_ray_step=72 --latent_mask=99 --inject_code_seed=890234

chair：
python scripts/inference3d-chair.py  --exp_dir=depth/chair-newpose120-box30 --dataset_type=chair_seg_to_3dface --pigan_curriculum_type=chair --checkpoint_path=latest_model.pt --data_path=data/chair/sample/  --test_output_size=512 --pigan_infer_ray_step=72  --latent_mask=41 --inject_code_seed=890234  --render_videos

CUDA_VISIBLE_DEVICES=2 nohup python scripts/inference3d-chair.py  --exp_dir=depth/onlypigan-chair-pose60-box25-ray0.8-10 --dataset_type=chair_seg_to_3dface --pigan_curriculum_type=chair --checkpoint_path=./pretrained_models/onlypigan-chair/iteration_20000.pt --data_path=data/chair/sample/  --test_output_size=512 --pigan_infer_ray_step=72  --latent_mask=41 --inject_code_seed=890234  --render_videos >3.log 2>&1 &

Record
	fov	r	box	ray
chair	60	6.2	25	0.8-9.0
replica				

