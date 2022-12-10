dataset_paths = {
    'celebahq_train': 'data/CelebAMask-HQ/CelebA-HQ-img',
    'celebahq_test': 'data/CelebAMask-HQ/CelebA-HQ-img',
    'celebahq_train_segmentation': 'data/CelebAMask-HQ/masks',
    'celebahq_test_segmentation': 'data/CelebAMask-HQ/mask_samples',
    'catmask_train': 'data/CatMask/images',
    'catmask_test': 'data/CatMask/images',
    'catmask_train_segmentation': 'data/CatMask/masks',
    'catmask_test_segmentation': 'data/CatMask/mask_samples',
    'replica_train': 'data/Sequence_1/rgb',
    'replica_test': 'data/Sequence_1/rgb',
    'replica_train_segmentation': 'data/Sequence_1/semantic_class',
    'replica_test_segmentation': 'data/Sequence_1/semantic_class',
    'chair_train': 'data/chair/images',
    'chair_test': 'data/chair/images',
    'chair_train_segmentation': 'data/chair/labels',
    'chair_test_segmentation': 'data/chair/labels',
}

model_paths = {
    'pigan_celeba': 'pretrained_models/pigan-celeba-pretrained.pth',
    'pigan_cat': 'pretrained_models/pigan-cats2-pretrained.pth',
    'swin_tiny': 'pretrained_models/swin_tiny_patch4_window7_224.pth',
}
