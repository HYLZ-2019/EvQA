import torch
import os
import numpy as np
import cv2
from utils.data import data_sources
from data.v2v_core_esim import EventEmulator
from data.patch_trigger import VoxelPatchTrigger
from data.v2v_datasets import WebvidDatasetV2

import ffmpeg
#import av
import json
import glob

def log_uniform(minval, maxval):
	eps = 1e-3
	logmin = np.log(minval + eps)
	logmax = np.log(maxval + eps)
	logval = np.random.uniform(logmin, logmax)
	return np.exp(logval) - eps

def bgr_to_gray(img_stack):
	# Convert (N, H, W, 3) to (N, H, W)
	gray = np.dot(img_stack[..., :3], [0.5870, 0.1140, 0.2989])
	return gray.astype(np.uint8)

class SparseDataset(WebvidDatasetV2): 

	def load_configs(self, configs):
		self.FPS = configs.get("FPS", 30)

		self.L = configs.get("sequence_length", 40)
		step_size = configs.get("step_size", None)
		
		self.proba_pause_when_running = configs.get("proba_pause_when_running", 0.01)
		self.proba_pause_when_paused = configs.get("proba_pause_when_paused", 0.98)

		self.fixed_seed = configs.get("fixed_seed", None)

		self.crop_size = configs.get("crop_size", None)
		self.fixed_crop = configs.get("fixed_crop", False)
		self.random_flip = configs.get("random_flip", True)
		self.num_bins = configs.get("num_bins", 5)
		self.frames_per_bin = configs.get("frames_per_bin", 1)
		self.frames_per_img = self.num_bins * self.frames_per_bin
		self.frames_per_seq = self.num_bins * self.frames_per_bin * self.L
		# Notice: this does not get non-overlapping samples. Each sample only goes 1 / self.frames_per_img sequences forward.
		self.step_size = step_size if step_size is not None else self.frames_per_seq

		self.min_resize_scale = configs.get("min_resize_scale", 0)
		self.max_resize_scale = configs.get("max_resize_scale", 1.3)
		self.max_rotate_degrees = configs.get("max_rotate_degrees", 0)

		self.shake_frames = configs.get("shake_frames", 0)
		self.shake_std = configs.get("shake_std", 0)

		self.threshold_range = configs.get("threshold_range", [0.05, 2])
		self.max_thres_pos_neg_gap = configs.get("max_thres_pos_neg_gap", 1.5)
		self.base_noise_std_range = configs.get("base_noise_std_range", [0, 0.2])
		self.hot_pixel_fraction_range = configs.get("hot_pixel_fraction_range", [0, 0.001])
		self.hot_pixel_std_range = configs.get("hot_pixel_std_range", [0, 0.2])
		self.put_noise_external = configs.get("put_noise_external", False)
		self.scale_noise_strength = configs.get("scale_noise_strength", False)
		self.max_samples_per_shot = configs.get("max_samples_per_shot", 1)
		self.subsample_ratio = configs.get("subsample_ratio", 1)
		
		self.force_hwaccel = configs.get("force_hwaccel", False)
		self.video_reader = configs.get("video_reader", "ffmpeg")
		assert self.video_reader in ["ffmpeg", "opencv"]

		self.keep_top_percentile = configs.get("keep_top_percentile", 0.54)
		self.use_fixed_thresholds = configs.get("use_fixed_thresholds", False)

		self.data_source_name = configs.get("data_source_name", "reds")
		# The index of data_source_name in data_sources
		self.data_source_idx = data_sources.index(self.data_source_name)

		self.color_mode = configs.get("color_mode", "gray")
		assert self.color_mode in ["gray", "gray_in_bgr_out"]

		assert (self.L > 0)
		assert (self.step_size > 0)

		# Output N+1 frames (0, 5, 10, ..., L*5) instead of N (5, 10, ..., L*5). Used to calculate ground truth optical flow in forward_sequence.
		self.output_additional_frame = configs.get("output_additional_frame", False)
		
		# Output N+1 event voxels. Because ERAFT model needs evs[i-1, i] and evs[i, i+1] to calculate flow[i, i+1].
		self.output_additional_evs = configs.get("output_additional_evs", False)
		if self.output_additional_evs:
			self.frames_per_seq += self.frames_per_img

		# Degrade the video qualities for ablation studies, proving that bad data is not good for training.
		self.video_degrade = configs.get("video_degrade", None)
		assert self.video_degrade in [None, "subtitles", "dirtyshotcut", "hdr", "ldr"]
		self.degrade_ratio = configs.get("degrade_ratio", 0)
		
        # new 

		self.trigger_type = configs.get("trigger_type", "naive")
		assert self.trigger_type in [None, "naive", "bias", "merge"]

		self.trigger_rate = configs.get("trigger_rate", 0.1)

		self.bin_type = configs.get("bin_type", "time")
		assert self.bin_type in ["time", "events"]
		
		self.min_patch = configs.get("min_patch", 32)
		assert self.min_patch % 16 == 0 and self.min_patch >= 16, "min_patch must be a multiple of 16 and >= 16."

		self.patch_dim = configs.get("patch_dim", 2)
		assert self.min_patch % self.patch_dim == 0

		self.patch_per_time = configs.get("patch_per_time", 2)

	def __init__(self, dataset_path, configs):
		self.load_configs(configs)

		self.dataset_path = dataset_path
		self.video_list_file = configs.get("video_list_file")
		with open(self.video_list_file, "r") as f:
			lines = f.readlines()
			data = [line.strip() for line in lines]
			# Each line: {video_subpath} {video_framecount}
			self.video_list = [line.split(" ")[0] for line in data]
			self.video_framecounts = [int(line.split(" ")[1]) for line in data]

			# Only will be used when self.use_fixed_thresholds
			self.video_pos_thres = [float(line.split(" ")[2]) for line in data]
			self.video_neg_thres = [float(line.split(" ")[3]) for line in data]
	
		self.sample_video_name = []
		self.sample_begin_idx = []
		self.sample_L = []
		self.sample_pos_thres = []
		self.sample_neg_thres = []

		for video_idx, (video_pth, frame_cnt) in enumerate(zip(self.video_list, self.video_framecounts)):
			shot_samples = 0
			for i in range(0, frame_cnt-self.frames_per_seq-1, self.step_size):
				self.sample_video_name.append(video_pth)
				self.sample_begin_idx.append(i)
				self.sample_L.append(self.L)
				self.sample_pos_thres.append(self.video_pos_thres[video_idx])
				self.sample_neg_thres.append(self.video_neg_thres[video_idx])
				shot_samples += 1
				#print(f"Added sample: video {video_pth}, start frame {i}, L {self.L}")
				if shot_samples >= self.max_samples_per_shot:
					break

		self.sample_video_name = np.array(self.sample_video_name)
		self.sample_begin_idx = np.array(self.sample_begin_idx)
		self.sample_L = np.array(self.sample_L)

		actual_sample_cnt = int(len(self.sample_L) * self.subsample_ratio)
		self.sample_video_name = self.sample_video_name[:actual_sample_cnt]
		self.sample_begin_idx = self.sample_begin_idx[:actual_sample_cnt]
		self.sample_L = self.sample_L[:actual_sample_cnt]
		self.sample_pos_thres = self.sample_pos_thres[:actual_sample_cnt]
		self.sample_neg_thres = self.sample_neg_thres[:actual_sample_cnt]

		#print(len(self.sample_video_name), len(self.video_pos_thres), len(self.video_neg_thres))

	def __len__(self):
		return len(self.sample_video_name)

	def __getitem__(self, sample_idx):
		""" Returns a list containing synchronized events <-> frame pairs
			[e_{i-L} <-> I_{i-L},
			e_{i-L+1} <-> I_{i-L+1},
			...,
			e_{i-1} <-> I_{i-1},
			e_i <-> I_i]
		"""
		# print("Loading sample index:", sample_idx)
		if self.fixed_seed is not None:
			# Fix the random seed, so that the randomness only depends on the idx of the validation batch.
			# Save the previous seed and recover it at the end of this function, so that the randomness of training is not affected.
			old_random_state = np.random.get_state()
			np.random.seed(self.fixed_seed + sample_idx)

		video_name = self.sample_video_name[sample_idx]
		start_frame = self.sample_begin_idx[sample_idx]
		img_cnt = self.sample_L[sample_idx]

		video_path = os.path.join(self.dataset_path, video_name)

		if self.video_reader == "ffmpeg":
			# Get video information
			probe = ffmpeg.probe(video_path)
			video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
			vid_width = int(video_info['width'])
			vid_height = int(video_info['height']) # Only keep top 54% since the shutterstock watermark is in the lower-middle.
		elif self.video_reader == "opencv":
			cap = cv2.VideoCapture(video_path)
			vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			cap.release()
		
		# print("Video size:", vid_width, vid_height)

		if self.crop_size is not None:
			min_resize_scale = max(
				self.min_resize_scale,
				self.crop_size / int(vid_height * self.keep_top_percentile),
				self.crop_size / vid_width
			)
			max_resize_scale = max(self.max_resize_scale, min_resize_scale)
		else:
			raise NotImplementedError("crop_size must be provided for WebvidDataset.")
	
		resize_scale = np.random.uniform(min_resize_scale, max_resize_scale)
		crop_size_before_resize = int(self.crop_size / resize_scale)

		if self.fixed_crop:
			min_i = 0
			min_j = 0
		else:
			min_i = np.random.randint(0, int(vid_height*self.keep_top_percentile) - crop_size_before_resize + 1)
			min_j = np.random.randint(0, vid_width - crop_size_before_resize + 1)

		# Initialize flip flag
		flip = False
		if self.random_flip and np.random.rand() > 0.5:
			flip = True

		# Get pause sequence
		img_idxes = []
		idx = 0
		is_pause = False

		# The additional frames are used for generating additional events. output_additional_frame does not need them.
		additional_frames = self.frames_per_img if self.output_additional_evs else 0

		for _ in range(start_frame, start_frame + img_cnt * self.frames_per_img + 1 + additional_frames):
			img_idxes.append(idx)
			if is_pause and np.random.rand() > self.proba_pause_when_paused:
				is_pause = False
			elif not is_pause and np.random.rand() < self.proba_pause_when_running:
				is_pause = True
			if not is_pause:
				idx += 1
		true_img_cnt = idx + 1

		end_frame = start_frame + true_img_cnt
		# raw_imgs: list of (H, W, 3) or (H, W, 1) images.
		raw_imgs = self.read_video(video_path, start_frame, end_frame, crop_size_before_resize, min_i, min_j, flip)

		if self.video_degrade is not None and np.random.rand() < self.degrade_ratio:
			raw_imgs = self.degrade_video(raw_imgs)

		# all_imgs: (N, H, W, 3) or (N, H, W, 1) images.
		all_imgs = np.stack([raw_imgs[i] for i in img_idxes])

		if self.color_mode == "gray":
			gray_imgs = all_imgs[..., 0]  # (N, H, W)
		elif self.color_mode == "gray_in_bgr_out":
			gray_imgs = bgr_to_gray(all_imgs)  # (N, H, W)

		FPS = 24
		# (N, num_bins, H, W)
		pos_thres = self.sample_pos_thres[sample_idx] if self.use_fixed_thresholds else None
		neg_thres = self.sample_neg_thres[sample_idx] if self.use_fixed_thresholds else None
		
		''' Debug: save gray images
		try:
			debug_dir = os.path.join("debug_gray")
			os.makedirs(debug_dir, exist_ok=True)

			n_show = min(16, gray_imgs.shape[0])
			rows = int(np.ceil(n_show / 4))
			cols = min(4, n_show)
			Ht, Wt = gray_imgs.shape[1], gray_imgs.shape[2]
			grid = np.zeros((rows * Ht, cols * Wt), dtype=np.uint8)
			for k in range(n_show):
				r, c = divmod(k, cols)
				grid[r*Ht:(r+1)*Ht, c*Wt:(c+1)*Wt] = gray_imgs[k]
			cv2.imwrite(os.path.join(debug_dir, f"sample_{int(sample_idx)}_grid.png"), grid)

			out_mp4 = os.path.join(debug_dir, f"sample_{int(sample_idx)}.mp4")
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			vw = cv2.VideoWriter(out_mp4, fourcc, self.FPS, (Wt, Ht))
			for k in range(gray_imgs.shape[0]):
				frame_bgr = cv2.cvtColor(gray_imgs[k], cv2.COLOR_GRAY2BGR)
				vw.write(frame_bgr)
			vw.release()
		except Exception as e:
			print(f"[gray debug] failed to save preview: {e}")
		'''
		
		v2e_params, all_patches = self.imgs_to_voxels(sample_idx,gray_imgs, self.num_bins, self.frames_per_bin, FPS, pos_thres, neg_thres)

		img_patches = []
		all_imgs_padded = np.zeros((all_imgs.shape[0], v2e_params["padded_H"],v2e_params["padded_W"], all_imgs.shape[3]), dtype=all_imgs.dtype)
		all_imgs_padded[:, :all_imgs.shape[1], :all_imgs.shape[2], :] = all_imgs
		# select img_patches according to patches
		for patch in all_patches:
			ey = patch['ey']
			ex = patch['ex']
			eh = patch['eh']
			ew = patch['ew']
			end_time = patch['end_t']
			img = all_imgs_padded[end_time, ey:ey+eh, ex:ex+ew, :].astype(np.float32) / 255 # Ground truth should be in [0, 1]
			img_patches.append(img)
		#print("img_patchesshape:",img_patches[0].shape)
		# print("Number of patches:", len(all_patches))
		voxel_flat, gt_flat, ex_arr, ey_arr, eh_arr, ew_arr, begin_t_arr, end_t_arr, begin_idx_arr = self.pack_dataset(all_patches,img_patches)
		'''
		print("voxel_flat shape:", voxel_flat.shape if voxel_flat is not None else None)
		print("gt_flat shape:", gt_flat.shape if gt_flat is not None else None)
		print("begin_t_arr:", begin_t_arr.shape)
		print("end_t_arr:", end_t_arr.shape)
		print("ex_arr:", ex_arr.shape)
		print("begin_idx_arr:", begin_idx_arr.shape)
		'''
		if (voxel_flat is not None):
			begin_t_arr = begin_t_arr/self.FPS
			end_t_arr = end_t_arr/self.FPS

		sequence = {
			"H": v2e_params["padded_H"],
			"W": v2e_params["padded_W"],
			"voxel_bins": self.num_bins,
			"sequence_name": f"seq_{idx}",
			"data_source_name": self.data_source_name,
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

		# 在训练循环中，获取batch后立即检查
		if np.isnan(voxel_flat).any():
			print(f"NaN detected in voxel at index {sample_idx}")
			print(f"Voxel stats: min={voxel_flat.min()}, max={voxel_flat.max()}, mean={voxel_flat.mean()}")
			assert False, f"NaN detected in voxel at index {sample_idx}"
		
		if np.isinf(voxel_flat).any():
			print(f"Inf detected in voxel at index {sample_idx}")
			assert False, f"Inf detected in voxel at index {sample_idx}"
		
		assert (voxel_flat.max() < 1e8) and (voxel_flat.min() > -1e8), f"Voxel values out of expected range at index {sample_idx}: min={voxel_flat.min()}, max={voxel_flat.max()}"

		if self.fixed_seed is not None:
			np.random.set_state(old_random_state)

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

		'''DEBUG'''
		n = len(gt_list)
		# assert n%2 == 0, f"Number of patches should be even, got {n}."
		# for i in range(0,n//2,2):
		# 	ex,ey,eh,ew = ex_list[i], ey_list[i], eh_list[i], ew_list[i]
		# 	ex2,ey2,eh2,ew2 = ex_list[i+1], ey_list[i+1], eh_list[i+1], ew_list[i+1]
		# 	assert (ex == ex2) and (ey == ey2) and (eh == eh2) and (ew == ew2), f"Paired patches should have the same coordinates, got patch {i} and {i+1}."
		'''DEBUG END'''	
		for gt_flat in gt_list:
			begin_idx_list.append(cur_idx)
			cur_idx += gt_flat.shape[0] # idx of voxels are num_bins times that of gt
		begin_idx_arr = np.array(begin_idx_list)
		voxel_flat = torch.cat(voxel_list, dim=0)
		gt_flat = torch.cat(gt_list, dim=0)
		return voxel_flat, gt_flat, ex_list, ey_list, eh_list, ew_list, begin_t_list, end_t_list, begin_idx_arr


	def imgs_to_voxels(self, sample_idx, imgs, num_bins, frames_per_bin, FPS, pos_thres=None, neg_thres=None):
		N, H, W = imgs.shape
		# print("Converting imgs to voxels:", imgs.shape,num_bins, frames_per_bin, FPS, pos_thres, neg_thres)
		assert (N-1) % (num_bins * frames_per_bin) == 0
		frame_cnt = (N-1) // (num_bins * frames_per_bin)

		if not self.use_fixed_thresholds: # Use random thresholds. use_fixed_thresholds is for ablation training.
			thres_1 = np.random.uniform(*self.threshold_range)
			pos_neg_gap = np.random.uniform(1, self.max_thres_pos_neg_gap)
			thres_2 = thres_1 * pos_neg_gap
			if np.random.rand() > 0.5:
				pos_thres = thres_1
				neg_thres = thres_2
			else:
				pos_thres = thres_2
				neg_thres = thres_1

		base_noise_std = np.random.uniform(*self.base_noise_std_range)
		hot_pixel_fraction = np.random.uniform(*self.hot_pixel_fraction_range)
		hot_pixel_std = np.random.uniform(*self.hot_pixel_std_range)

		if self.scale_noise_strength and not self.put_noise_external:
			# The same base_noise_std should lead to the same amount of pure noise events, independent to the threshold.
			base_noise_std = base_noise_std * pos_thres
			hot_pixel_std = hot_pixel_std * pos_thres

		all_voxels = EventEmulator(
			pos_thres=pos_thres,
			neg_thres=neg_thres,
			base_noise_std=base_noise_std,
			hot_pixel_fraction=hot_pixel_fraction,
			hot_pixel_std=hot_pixel_std,
			put_noise_external=self.put_noise_external,
			seed = None
		).video_to_voxel(imgs)
		'''
		#Debug: 保存 all_voxels（压缩 .npz）
		try:
			save_dir = "debug_voxels"
			os.makedirs(save_dir, exist_ok=True)
			vox_np = all_voxels if isinstance(all_voxels, np.ndarray) else all_voxels.detach().cpu().numpy()
			out_path = os.path.join(save_dir, f"vox_{sample_idx}_{os.getpid()}.npz")
			np.savez_compressed(
				out_path,
				voxels=vox_np,
				FPS=FPS,
				num_bins=num_bins,
				frames_per_bin=frames_per_bin,
				pos_thres=pos_thres,
				neg_thres=neg_thres
			)
			# 可选：打印一次路径
			# print(f"[vox debug] saved to {out_path}, shape={vox_np.shape}")
		except Exception as e:
			print(f"[vox debug] save failed: {e}")
		'''

		# Reshape to (N, num_bins, frames_per_bin, H, W)
		all_patches, padded_H, padded_W = VoxelPatchTrigger(
			trigger_type=self.trigger_type,
			num_bins=num_bins,
			bin_type=self.bin_type,
			patch_size=self.min_patch,
			trigger_rate=self.trigger_rate,
			patch_dim = self.patch_dim,
			patch_per_time=self.patch_per_time,
		).process(all_voxels)

		v2e_params = {
			"padded_H": padded_H,
			"padded_W": padded_W,
			"pos_thres": pos_thres,
			"neg_thres": neg_thres,
			"base_noise_std": base_noise_std,
			"hot_pixel_fraction": hot_pixel_fraction,
			"hot_pixel_std": hot_pixel_std,
		}

		return v2e_params, all_patches

def no_collate_function(batch):
	"""
    自定义的 collate 函数来处理不同大小的矩阵。
    'batch' 是一个列表，其长度等于 batch_size。
    batch 中的每个元素都是 CustomDataset 的 __getitem__ 的返回值。
    例如, 如果 batch_size 是 2, batch 可能是:
    [
        [torch.randn(10, 20), torch.randn(5, 5)],  # 第一个样本
        [torch.randn(12, 18), torch.randn(6, 7)]   # 第二个样本
    ]
    """
	return batch[0]