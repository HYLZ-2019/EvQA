import torch
import h5py
import os
import numpy as np
import cv2
from data.patch_trigger import EventPatchTrigger

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import data_sources

# The authors of HQF have already cut the time ranges.

class SparseTestH5Dataset(torch.utils.data.Dataset):
    '''
    Test with frames from HQF-like format.
    '''
    def __init__(self, h5_path, configs):
        # 传入的是单个testdataset的config
        self.h5_path = h5_path
        # The h5_path should be like HQF_h5/bike_bay_hdr.h5. The corresponding sequence_name is hqf_bike_bay_hdr.
        self.sequence_name = os.path.basename(h5_path).split(".")[0]

        self.configs = configs
        # 它属于EvQA里的哪个子集
        self.dataset_name = os.path.basename(os.path.dirname(h5_path))
        self.num_bins = configs.get("num_bins", 5)
        self.need_two_frame = configs.get("need_two_frame", False)

        self.trigger_type = configs.get("trigger_type", "naive")
        assert self.trigger_type in [None, "naive", "bias", "merge"]

        self.trigger_rate = configs.get("trigger_rate", 0.1)

        self.bin_type = configs.get("bin_type", "time")
        assert self.bin_type in ["time", "events"]

        self.min_patch = configs.get("min_patch", 32)
        assert self.min_patch % 16 == 0 and self.min_patch >= 16, "min_patch must be a multiple of 16 and >= 16."

        self.event_batch_size = configs.get("event_batch_size", 50000)
        self.patch_per_time = configs.get("patch_per_time", 2)
        self.patch_dim = configs.get("patch_dim", 2)
        self.trigger_param_gen_name = configs.get("trigger_param_gen_name", "default")

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        #print(f"Processing sequence: {self.sequence_name}, index: {idx}")
        all_patches, padded_H, padded_W = EventPatchTrigger(
            trigger_type = self.trigger_type,
            num_bins=self.num_bins, 
            bin_type=self.bin_type,
            patch_size=self.min_patch,
            trigger_rate=self.trigger_rate,
            event_batch_size=self.event_batch_size,
            patch_per_time=self.patch_per_time,
            patch_dim = self.patch_dim,
            trigger_param_gen_name=self.trigger_param_gen_name,
		).process(self.h5_path)

        img_patches = []
        # select img_patches according to patches
        for patch in all_patches:
            eh = patch['eh']
            ew = patch['ew']
            img = np.zeros((eh, ew,1), dtype=np.float32)
            img_patches.append(img)
        # print("img_patchesshape:",img_patches[0].shape)
        voxel_flat, gt_flat, ex_arr, ey_arr, eh_arr, ew_arr, begin_t_arr, end_t_arr, begin_idx_arr = self.pack_dataset(all_patches,img_patches)
        if (voxel_flat is not None):
            begin_t_arr = begin_t_arr/1e6
            end_t_arr = end_t_arr/1e6
        '''
        print("voxel_flat shape:", voxel_flat.shape if voxel_flat is not None else None)
        print("gt_flat shape:", gt_flat.shape if gt_flat is not None else None)
        print("begin_t_arr:", begin_t_arr.shape)
        print("end_t_arr:", end_t_arr.shape)
        print("ex_arr:", ex_arr.shape)
        print("begin_idx_arr:", begin_idx_arr.shape)
        '''
        sequence = {
			"H": padded_H,
			"W": padded_W,
			"voxel_bins": self.num_bins,
			"sequence_name": self.sequence_name,
			"dataset_name": self.dataset_name,
			"voxel_flat": voxel_flat, # (N, 5, H, W)
			"gt_flat": gt_flat, # (N, 1, H, W)
			"ex_arr": ex_arr, # (N,)
			"ey_arr": ey_arr, # (N,)
			"eh_arr": eh_arr, # (N,)
			"ew_arr": ew_arr, # (N,)
			"begin_t_arr": begin_t_arr, # (N,)
			"end_t_arr": end_t_arr, # (N,)
			"begin_idx_arr": begin_idx_arr, # (N,)
		}
        # print (f"Total patches in sequence {self.sequence_name}: {len(all_patches)}")
        # print("----",sequence["voxel_bins"])
        return sequence

    def pack_dataset(self, patch_list, img_patch_list):
        """
        Pack a list of patches into flat tensors and arrays.
        Args:
            patch_list: list of patches, each patch is a dict with keys:
                'voxel': (5, H, W) tensor
                'gt_img': (1, H, W) tensor
                'ex': int
                'ey': int
                'eh': int
                'ew': int
                'begin_t': float
                'end_t': float
        Returns:
            voxel_flat: (N, 5, H, W) tensor
            gt_flat: (N, 1, H, W) tensor
            ex_arr: (N,) array
            ey_arr: (N,) array
            eh_arr: (N,) array
            ew_arr: (N,) array
            begin_t_arr: (N,) array
            end_t_arr: (N,) array
            begin_idx_arr: (N,) array
		"""
        if (patch_list is None) or (len(patch_list) == 0):
            return None, None, None, None, None, None, None, None, None
		# Patches have different size; flatten all of them, concat them into voxel_flat, and save the beginning index of each patch in begin_idx_arr. gt_flat is the ground truth image for each patch.
        voxel_list = [torch.from_numpy(obj['voxel'].flatten()) for obj in patch_list]
        gt_list = [torch.from_numpy(obj.flatten()) for obj in img_patch_list]
        ex_list = np.array([obj['ex'] for obj in patch_list])
        ey_list = np.array([obj['ey'] for obj in patch_list])
        eh_list = np.array([obj['eh'] for obj in patch_list])
        ew_list = np.array([obj['ew'] for obj in patch_list])
        begin_t_list = np.array([obj['begin_t'] for obj in patch_list])
        end_t_list = np.array([obj['end_t'] for obj in patch_list])
        begin_idx_list = []
        cur_idx = 0
        for gt_flat in gt_list:
            begin_idx_list.append(cur_idx)
            cur_idx += gt_flat.shape[0] # idx of voxels are num_bins times that of gt
        begin_idx_arr = np.array(begin_idx_list)
        voxel_flat = torch.cat(voxel_list, dim=0)
        gt_flat = torch.cat(gt_list, dim=0)
        return voxel_flat, gt_flat, ex_list, ey_list, eh_list, ew_list, begin_t_list, end_t_list, begin_idx_arr

