from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
    'celebahq_seg_to_3dface': {
        'transforms': transforms_config.SegToImageTransforms3D,
        'train_source_root': dataset_paths['celebahq_train_segmentation'],
        'train_target_root': dataset_paths['celebahq_train'],
        'test_source_root': dataset_paths['celebahq_test_segmentation'],
        'test_target_root': dataset_paths['celebahq_test'],
    },
    'catmask_seg_to_3dface': {
        'transforms': transforms_config.SegToCatTransforms3D,
        'train_source_root': dataset_paths['catmask_train_segmentation'],
        'train_target_root': dataset_paths['catmask_train'],
        'test_source_root': dataset_paths['catmask_test_segmentation'],
        'test_target_root': dataset_paths['catmask_test'],
    },
    'replica_seg_to_3dface': {
        'transforms': transforms_config.SegToCatTransforms3D,
        # 'train_source_root': dataset_paths['replica_train'],
        'train_source_root': dataset_paths['replica_train_segmentation'],
        'train_target_root': dataset_paths['replica_train'],
        # 'test_source_root': dataset_paths['replica_test'],
        'test_source_root': dataset_paths['replica_test_segmentation'],
        'test_target_root': dataset_paths['replica_test'],
    },
    'chair_seg_to_3dface': {
        'transforms': transforms_config.SegToCatTransforms3D,
        'train_source_root': dataset_paths['chair_train_segmentation'],
        'train_target_root': dataset_paths['chair_train'],
        'test_source_root': dataset_paths['chair_test_segmentation'],
        'test_target_root': dataset_paths['chair_test'],
    }
}
