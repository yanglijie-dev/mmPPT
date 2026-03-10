"""
mmBody Dataset

Author: Lijie Yang
"""
import torch
import os
import numpy as np
from collections.abc import Sequence
import pickle
import random
from .builder import DATASETS
from .defaults import DefaultDataset
from torch.utils.data import Dataset
from .transform import Compose, TRANSFORMS
class mmBodySequenceLoader(object):
    def __init__(self, seq_path: str, skip_head: int = 0, skip_tail: int = 0, resource=['radar']) -> None:
        self.seq_path = seq_path
        self.skip_head = skip_head
        self.skip_tail = skip_tail
        self.resource = resource
        # load transformation matrix
        with open(os.path.join(seq_path, 'calib.txt')) as f:#load sensor calibration parameters
            calib = eval(f.readline())
        self.calib = {
            'image':calib['kinect_master'],
            'depth':calib['kinect_master'],
        }

    def __len__(self):
        return len(os.listdir(os.path.join(self.seq_path, 'mesh'))) - self.skip_head - self.skip_tail

    def __getitem__(self, idx: int):
        result = {}
        if 'radar' in self.resource:
#            print(os.path.join(self.seq_path, 'radar', 'frame_{}.npy'.format(idx+self.skip_head)))
            result['radar'] = np.load(os.path.join(
                self.seq_path, 'radar', 'frame_{}.npy'.format(idx+self.skip_head)))
        if 'image' in self.resource:
            result['image'] = cv2.imread(os.path.join(
                self.seq_path, 'image', 'master', 'frame_{}.png'.format(idx+self.skip_head)))
        if 'depth' in self.resource:
            result['depth'] = np.load(os.path.join(
                self.seq_path, 'depth_pcl', 'master', 'frame_{}.npy'.format(idx+self.skip_head)))
        result['mesh'] = np.load(os.path.join(
            self.seq_path, 'mesh', 'frame_{}.npz'.format(idx+self.skip_head)))
        #print('Current reading file is ',os.path.join(self.seq_path, 'radar', 'frame_{}.npy'.format(idx+self.skip_head)))
        return result

@DATASETS.register_module()
class mmBodyDataset(Dataset):
    def __init__(self, **kwargs):
        self.data_path = kwargs.get('data_path')
        self.train = kwargs.get('train')
        self.clip_step = kwargs.get('clip_step', 1)
        self.clip_frames = kwargs.get('clip_frames')[0]
        self.clip_range = self.clip_frames * self.clip_step

        self.output_dim = kwargs.get('output_dim')[0]
        self.skip_head = kwargs.get('skip_head')[0]
        self.skip_tail = kwargs.get('skip_tail', 0)
        self.test_scene = kwargs.get('test_scene')[0]
        self.input_data = kwargs.get('input_data')[0]
        self.num_points = kwargs.get('num_points')[0]
        self.seq_idxes = kwargs.get('seq_idxes')[0]
        self.features = kwargs.get('feat_dim')[0]
        self.transform = Compose(kwargs.get('transform'))

        self.init_index_map()

        self.point_shuffle = kwargs.get('point_shuffle')


    def init_index_map(self):
        # init the index map for each frame
        self.index_map = [0, ]
        if self.train:
            seq_dirs = ['sequence_{}'.format(i) for i in self.seq_idxes]  # sequence_0~19
            self.seq_paths = [os.path.join(self.data_path, "train", p) for p in seq_dirs]  # ./dataset/mmBody/train/sequence_0~19
        else:
            seq_dirs = ['sequence_{}'.format(i) for i in range(2)]#each test scene contains two sequences
            self.seq_paths = [os.path.join(self.data_path, self.test_scene, p) for p in seq_dirs]

        print('Data path: ', self.seq_paths)
        self.seq_loaders = {}
        for path in self.seq_paths:
            seq_loader = mmBodySequenceLoader(path, self.skip_head, self.skip_tail, resource=self.input_data)
            self.seq_loaders.update({path: seq_loader})
            self.index_map.append(self.index_map[-1] + len(seq_loader))

    def global_to_seq_index(self, global_idx: int):
        for seq_idx in range(len(self.index_map) - 1):
            if global_idx in range(self.index_map[seq_idx], self.index_map[seq_idx + 1]):
                frame_idx = global_idx - self.index_map[seq_idx]
                # print('frame_idx = ',frame_idx)
                return seq_idx, frame_idx
            # print('seq_idx = ', seq_idx)
        # raise IndexError

    def pad_data(self, data, return_choices=False):
        # pad point cloud with the fixed num of points
        if data.shape[0] > self.num_points:
            r = np.random.choice(data.shape[0], size=self.num_points, replace=False)
        else:
            repeat, residue = self.num_points // data.shape[0], self.num_points % data.shape[0]
            r = np.random.choice(data.shape[0], size=residue, replace=False)
            r = np.concatenate([np.arange(data.shape[0]) for _ in range(repeat)] + [r], axis=0)
        if return_choices:
            return data[r, :], r
        return data[r, :]

    def filter_pcl(self, bounding_pcl: np.ndarray, target_pcl: np.ndarray, bound: float = 0.2, offset: float = 0):
        """
        Filter out the pcls of pcl_b that is not in the bounding_box of pcl_a
        """
        upper_bound = bounding_pcl[:, :3].max(axis=0) + bound
        lower_bound = bounding_pcl[:, :3].min(axis=0) - bound
        lower_bound[2] += offset

        mask_x = (target_pcl[:, 0] >= lower_bound[0]) & (
                target_pcl[:, 0] <= upper_bound[0])
        mask_y = (target_pcl[:, 1] >= lower_bound[1]) & (
                target_pcl[:, 1] <= upper_bound[1])
        mask_z = (target_pcl[:, 2] >= lower_bound[2]) & (
                target_pcl[:, 2] <= upper_bound[2])
        index = mask_x & mask_y & mask_z
        return target_pcl[index]

    def load_data(self, seq_loader, idx):
        # print('idx = ',idx)
        frame = seq_loader[idx]
        radar_pcl = frame['radar']
        radar_pcl[:, 3:] /= np.array([5e-38, 5., 150.])

        mesh_pose = frame['mesh']['pose']
        mesh_shape = frame['mesh']['shape']
        mesh_joint = frame['mesh']['joints'][:22]
        arbe_data = self.filter_pcl(mesh_joint, radar_pcl, 0.2)

        if arbe_data.shape[0] == 0:
            # remove bad frame
            return None, None

        bbox_center = ((mesh_joint.max(axis=0) + mesh_joint.min(axis=0)) / 2)[:3]
        arbe_data[:, :3] -= bbox_center
        mesh_pose[:3] -= bbox_center

        # padding
        arbe_data = self.pad_data(arbe_data)
        label = np.concatenate((mesh_pose, mesh_shape), axis=0)

        return arbe_data, label

    def __len__(self):
        return self.index_map[-1]

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.global_to_seq_index(idx)
        seq_path = self.seq_paths[seq_idx]
        clip = []
        seq_loader = self.seq_loaders[seq_path]
        data, label = self.load_data(seq_loader,
                                     frame_idx)
        while data is None:
            frame_idx = random.randint(self.clip_range - 1, len(seq_loader) - 1)
            data, label = self.load_data(seq_loader, frame_idx)

        t = np.zeros((data.shape[0], 1), dtype=data.dtype)
        T = [t]
        for clip_id in range(frame_idx - self.clip_range + 1, frame_idx, self.clip_step):
            # print('clip_id = ',clip_id)
            # get xyz and features
            clip_data, _ = self.load_data(seq_loader, clip_id)
            # remove bad frame
            if clip_data is None:
                clip_data = data
            # padding
            clip.append(clip_data)
            T.append(T[-1] + 1)
        clip.append(data)

        T = np.stack(T, axis=0)

        clip = np.asarray(clip, dtype=np.float32)
        label = np.asarray(label, dtype=np.float32)
        assert clip.shape[-1]==6
        clip = np.concatenate((clip, T), axis=-1)

        for i in range(3,clip.shape[-1]-1):
            clip[:, :, i] -= clip[:, :, i].min()
            clip[:, :, i] /= clip[:, :, i].max()

        if True in np.isnan(label):
            label = np.nan_to_num(label)

        data_dict = dict(
            coord = clip[:,:,:3],
            features = clip[:,:,3:-1],
            label = label
        )

        data_dict['coord'] = data_dict['coord'].reshape(-1, 3)
        data_dict['features'] = data_dict['features'].reshape(-1, 3)
        data_dict = self.transform(data_dict)

        data_dict['file_id']="seq_idx_"+str(seq_idx)+"_frame_idx_"+str(frame_idx+self.skip_head)

        data_dict['temporal_info'] = clip[:,:,-1]

        if self.point_shuffle==True:#We also need to shuffle point clouds in each bath for "xyz" to remove the latent temporal info inside point clouds
            points_per_batch = data_dict["coord"].shape[0]
            idx_point_shuffle = list(range(points_per_batch))
            random.shuffle(idx_point_shuffle)
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor) and data_dict[key].shape[0] == points_per_batch:
                    data_dict[key] = data_dict[key][idx_point_shuffle]
        return data_dict