from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import os
import numpy as np
import torchvision.transforms as transforms
import torch
from utils.data_utils import mask_to_binary_masks, binary_masks_to_contour, contour_to_dist
import re

class InferenceDataset3D(Dataset):

    def __init__(self, root, opts, transform=None, paths_conf=None):
        if paths_conf is None:
            self.paths = sorted(data_utils.make_dataset(root))
        else:
            with open(paths_conf, 'r') as f:
                paths = f.readlines()
                paths = [x.strip() for x in paths if x.strip()]
            src_ext = os.path.splitext(os.listdir(root)[0])[-1]
            self.paths = sorted([os.path.join(root, "%s%s" % (x, src_ext)) for x in paths])

        self.transform = transform
        self.opts = opts
        self.env_name = opts.pigan_curriculum_type
        if self.env_name == "CelebAMask_HQ":
            pose_file_name = "CelebAMask-HQ-pose-anno.txt"
        elif self.env_name == "CatMask":
            pose_file_name = "pose-pseudo.txt"
        elif self.env_name == "replica":
            pose_file_name = "traj_w_c.txt"
        elif self.env_name == "chair":
            pose_file_name = "traj_w_c.txt"
        else:
            raise Exception("Cannot find environment %s" % self.env_name)
        pose_path = os.path.join(*root.split('/')[:-1], pose_file_name)
        # if opts.use_original_pose:
            # self.pose_dict = self.build_poses_mapping_dict(pose_path)
        
        if opts.use_contour:
            self.contour_transfrom = transforms.Compose(self.transform.transforms[:-2] + [self.transform.transforms[-1]])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from_path = self.paths[index]
        # from_im = Image.open(from_path)
        # from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')
        # from_im = self.update_labels(from_im)
        # from_im_raw = from_im
        # if self.transform:
        #     from_im = self.transform(from_im)

        from_ims = []
        for filename in os.listdir('/home/chenlinsheng/sem2nerf/data/Sequence_1/semantic-masks'):
            # from_path_1 = self.source_paths[index]
            # print(index)
            from_im = Image.open('/home/chenlinsheng/sem2nerf/data/Sequence_1/semantic-masks/' + filename)
            from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')
            from_im = self.update_labels(from_im)
            from_im_raw = from_im
            if self.source_transform:
                from_im = self.source_transform(from_im)
            from_ims.append(from_im)

        im_id = os.path.splitext(os.path.basename(from_path))[0]

        # create augmented data on-the-fly if needed
        if self.opts.use_contour:
            # create contour tensor
            in_mask_np = mask_to_binary_masks(from_im_raw)
            from_contour = binary_masks_to_contour(in_mask_np)
            contour_tensor = self.contour_transfrom(from_contour)

            # create distance data
            dist = contour_to_dist(from_contour)
            dist_tensor = self.contour_transfrom(dist)

            from_im = torch.cat([from_im, contour_tensor, dist_tensor], dim=0)

        if self.opts.use_original_pose:
            # im_pose = self.get_pose_by_image_path(from_path)
            im_pose = self.get_pose(from_path)
            return from_ims, im_pose, im_id
        else:
            im_pose = self.get_pose(from_path)
            return from_ims, im_pose, im_id

    def build_poses_mapping_dict(self, poses_file):
        if self.env_name == "CelebAMask_HQ":
            n_headlines = 2
            n_pose_params = 3
        elif self.env_name == "CatMask":
            n_headlines = 1
            n_pose_params = 2
        else:
            raise Exception("Cannot find environment %s" % self.env_name)

        with open(poses_file, 'r') as f:
            lines = f.readlines()
            lines = [x for x in lines if x.strip()][n_headlines:]
        poses_dict = {}
        for line in lines:
            items = line.split(' ')
            cur_id = int(items[0].split('.')[0])
            poses_dict[cur_id] = [float(x) for x in items[1:n_pose_params+1]]

        return poses_dict

    def get_pose_by_image_path(self, image_path):
        image_name = os.path.basename(image_path).split('.')[0]
        cur_id = int(image_name.split('_')[0])
        cur_pose = self.pose_dict[cur_id]
        if self.env_name == "CelebAMask_HQ":
            yaw = np.deg2rad(90 - cur_pose[0])
            pitch = np.deg2rad(90 + cur_pose[1])
        elif self.env_name == "CatMask":
            yaw = cur_pose[0]
            pitch = cur_pose[1]
        else:
            raise Exception("Cannot find environment %s" % self.env_name)
        return yaw, pitch
    
    def get_pose(self, image_path):
        # cur_id = 0
        cur_id = int(os.path.basename(image_path).split('.')[0])
        
        with open('/home/chenlinsheng/sem2nerf/data/Sequence_1/traj_w_c.txt') as file:
            p = file.read()

        rows = p.split('\n')
        pose = []
        for row in rows:
            pose.append(re.findall('[\d+-\.e]+', row))
        # np.array(pose, dtype=float)
        cur_pose = pose[cur_id]
        cur_pose = np.array(cur_pose).reshape(4,4)
        cur_pose = cur_pose.astype(np.float32)

        return cur_pose

    def update_labels(self, ori_labels):
        if not self.opts.use_merged_labels:
            return ori_labels

        if self.env_name == "CelebAMask_HQ":
            new_labels = [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        elif self.env_name == "CatMask":
            new_labels = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 7, 0]
        elif self.env_name == "replica":
            new_labels = [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
                        10, 11, 12, 13, 14, 14, 15, 15, 16, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 
                        60, 61, 62, 63, 64, 64, 65, 65, 66, 66, 67, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                        70, 31, 32, 33, 34, 34, 35, 35, 36, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                        40, 41, 42, 43, 44, 44, 45, 45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
        elif self.env_name == "chair":
            new_labels = [0, 1, 2, 3, 4, 44, 5, 45, 46, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
                        20, 21, 22, 23, 24, 24, 25, 25, 26, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
        else:
            raise Exception("Cannot find environment %s" % self.env_name)

        labels_np = np.array(ori_labels)
        for idx, label in enumerate(new_labels):
            labels_np[labels_np == idx] = label
        new_labels = Image.fromarray(np.uint8(labels_np))
        return new_labels
