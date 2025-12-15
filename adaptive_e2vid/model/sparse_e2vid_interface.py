import torch
import torchmetrics
import torch.nn.functional as F

import traceback
import collections
import numpy as np
from skimage.metrics import structural_similarity as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from collections import defaultdict


from utils.util import instantiate_from_config
from model.loss import l2_loss, perceptual_loss, l1_loss, ssim_loss, temporal_consistency_loss
# Names of each data source: ('esim', 'ijrr', ...)
from utils.data import data_sources
from PerceptualSimilarity.models import PerceptualLoss
from torchvision.models.optical_flow import raft_small, raft_large
import matplotlib.pyplot as plt

from model.train_utils import load_raft, inference_raft, norm, printshapes, nan_hook, normalize, concat_imgs, normalize_nobias, normalize_batch_voxel

def unpack_dataset(obj):
	sequence = []
	patch_num = obj['begin_idx_arr'].shape[0]
	voxel_bins = obj["voxel_bins"]
	for i in range(patch_num):
		begin_idx = obj['begin_idx_arr'][i]
		end_idx = obj['begin_idx_arr'][i+1] if i < patch_num - 1 else obj['voxel_flat'].shape[0]
		sequence.append({
			'voxel': obj['voxel_flat'][begin_idx*voxel_bins:end_idx*voxel_bins].reshape(obj["voxel_bins"], obj['eh_arr'][i], obj['ew_arr'][i]),
			'gt_img': obj['gt_flat'][begin_idx:end_idx].reshape(1, obj['eh_arr'][i], obj['ew_arr'][i]),
			'ex': obj['ex_arr'][i],
			'ey': obj['ey_arr'][i],
			'eh': obj['eh_arr'][i],
			'ew': obj['ew_arr'][i],
			'begin_t': obj['begin_t_arr'][i],
			'end_t': obj['end_t_arr'][i]
		})
	return sequence

class SparseE2VIDInterface():
	def __init__(self, configs, device, local_rank):
		
		self.configs = configs
		self.device = device
		self.local_rank = local_rank

		self.e2vid_model = instantiate_from_config(configs["model"]).to(device)

		for submodule in self.e2vid_model.modules():
			submodule.register_forward_hook(nan_hook)

		# RAFT is used for generation of optical flow when real optical flow is not available and we also need to calculate the temporal consistency loss.
		# Else, don't bother to load it.
		if configs["loss"].get("temporal_consistency_weight", 0) > 0:
			self.optical_flow_source = configs["loss"].get("optical_flow_source", "gt")
			assert self.optical_flow_source in ["raft_small", "raft_large", "gt", "zeros"], f"Unknown optical flow source: {self.optical_flow_source}"
	
			if self.optical_flow_source == "raft_small":
				self.raft_model = load_raft("raft_small", device=device)
			elif self.optical_flow_source == "raft_large":
				self.raft_model = load_raft("raft_large", device=device)

			self.raft_num_flow_updates = configs["loss"].get("raft_num_flow_updates", 12)

		self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=255)
		self.loss_functions = self.load_loss_functions(configs["loss"])

		# Not the most modern LPIPS model; used to keep metric consistency with previous papers
		self.test_lpips_fn = PerceptualLoss(net="alex")

		self.current_epoch = 0
		self.pred_channels = configs.get("pred_channels", 1)


	def set_current_epoch(self, epoch):
		self.current_epoch = epoch
		
	def compute_metrics(self, pred, batch):
		# Calculate metrics for test.
		# Pred is list of predicted patches.
		data = batch # 由于特殊的no_collate_function，batch本身就是一个样本
		H = data["H"]
		W = data["W"]
		T = data["begin_idx_arr"].shape[0]
		patches = unpack_dataset(data)

		data_source_name = data["data_source_name"]
		sequence_name = data["sequence_name"]
		log_prefix = f"{data_source_name.upper()}/{sequence_name}"
		
		metrics = defaultdict(list)

		for t in range(T):
			# Calculate MSE, SSIM and LPIPS.
			# Use exactly the same calculating method as in ET-Net code.
			# ET-Net uses skimage.metrics.structural_similarity for SSIM, which has default window size 7. The torchvision version has default window size 11.
			# The metrics are calculated in the [0, 1] range.
			pred_image = pred[0, t]
			gt = patches[t]

			mse = F.mse_loss(pred_image, gt).item()
			lpips = self.test_lpips_fn(pred_image, gt, normalize=True).item()

			pred_image = pred_image.detach().cpu().numpy().squeeze()
			gt = gt.detach().cpu().numpy().squeeze()

			# According to the notes in https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity, we should set data_range=1 to SSIM because the input values are in [0, 1], while the data_range will be defaulted to 2 ([-1, 1]) if not set. However, the metrics in the previous papers (e.g. ET-Net) were calculated with the wrong data_range, so we'll follow them to keep consistency.
			ssim = SSIM(pred_image, gt, data_range=2)
			
			metrics[f"{log_prefix}/MSE"].append(mse)
			metrics[f"{log_prefix}/LPIPS"].append(lpips)
			metrics[f"{log_prefix}/SSIM"].append(ssim)

		return metrics

	def load_loss_functions(self, cfgs):
		funcs = []
		if cfgs.get("lpips_weight", 0) != 0:
			lpips_weight = cfgs["lpips_weight"]
			self.perceptual_loss = perceptual_loss(weight=lpips_weight, net=cfgs["lpips_type"])
			funcs.append(
				self.perceptual_loss
			)
		if cfgs.get("l2_weight", 0) != 0:
			l2_weight = cfgs["l2_weight"]
			funcs.append(
				l2_loss(weight=l2_weight)
			)
		if cfgs.get("l1_weight", 0) != 0:
			l1_weight = cfgs["l1_weight"]
			funcs.append(
				l1_loss(weight=l1_weight)
			)
		if cfgs.get("ssim_weight", 0) != 0:
			ssim_weight = cfgs["ssim_weight"]
			funcs.append(
				ssim_loss(weight=ssim_weight, model=self.ssim)
			)
		if cfgs.get("temporal_consistency_weight", 0) != 0:
			temporal_consistency_weight = cfgs["temporal_consistency_weight"]
			temporal_consistency_L0 = cfgs.get("temporal_consistency_L0", 1)
			funcs.append(
				temporal_consistency_loss(weight=temporal_consistency_weight, L0=temporal_consistency_L0)
			)
		return funcs
	
	def forward_sequence(self, batch, reset_states=True, test=False, val=False):
		# If there is no ground truth optical flow, predict some.
		# Testing does not require flow
		# Validation should have flow, but calculation of flow makes GPU OOM (because 720p EVAID x 80-img sequence is big)
		assert self.configs["loss"].get("temporal_consistency_weight", 0) == 0 , "Temporal consistency loss not supported yet."
		data = batch # 由于特殊的no_collate_function，batch本身就是一个样本
		H = data["H"]
		W = data["W"]
		T = data["begin_idx_arr"].shape[0]
		patches = unpack_dataset(data)
		device = data["voxel_flat"].device

		if reset_states:
			if self.local_rank is not None:
				self.e2vid_model.module.reset_states(B=1, H=H, W=W, device=device)
			else:
				self.e2vid_model.reset_states(B=1, H=H, W=W, device=device)

		pred_patches = []

		for t in range(T):
			patch = patches[t]
			pred = self.e2vid_model(
				event_tensor = patch["voxel"].unsqueeze(0),
				ey = patch["ey"],
				ex = patch["ex"],
				eh = patch["eh"],
				ew = patch["ew"],
				begin_t = patch["begin_t"],
				end_t = patch["end_t"]
			)
			pred_patches.append({
				"pred": pred,
				"ey": patch["ey"],
				"ex": patch["ex"],
				"eh": patch["eh"],
				"ew": patch["ew"]
			})
		
		# pred_imgs should be in [0, 1]
		return pred_patches
	
	def calc_loss(self, batch, pred, remove_flow_loss=False):
		data = batch # 由于特殊的no_collate_function，batch本身就是一个样本
		patches = unpack_dataset(data)
		data_source_name = data["data_source_name"]
		T = len(pred)
	
		losses = {}

		loss_functions_list = self.loss_functions

		for loss_ftn in loss_functions_list:
			loss_name = loss_ftn.__class__.__name__
			losses[loss_name] = torch.zeros((T), device=self.device)

		final_losses = collections.defaultdict(lambda: 0)

		for t in range(T):
			gt_img = patches[t]["gt_img"]
			pred_img = pred[t]["pred"]
			
			# Calculate the losses. 
			for loss_ftn in loss_functions_list:
				#start = time.time()
				loss_name = loss_ftn.__class__.__name__
				if isinstance(loss_ftn, perceptual_loss):
					ls = loss_ftn(pred_img, gt_img, reduce_batch=True)
				elif isinstance(loss_ftn, l2_loss):
					ls = loss_ftn(pred_img, gt_img, reduce_batch=True)
				elif isinstance(loss_ftn, l1_loss):
					ls = loss_ftn(pred_img, gt_img, reduce_batch=True)
				elif isinstance(loss_ftn, ssim_loss):
					raise NotImplementedError
				elif isinstance(loss_ftn, temporal_consistency_loss):
					raise NotImplementedError
				
				losses[loss_name][t] = ls
				#time_usage[loss_name] += time.time() - start
	
		for loss_ftn in loss_functions_list:
			loss_name = loss_ftn.__class__.__name__
			mean_loss = torch.mean(losses[loss_name])  # Average over T
			
			final_losses[f"{loss_name}/{data_source_name}"] += mean_loss.item()
			final_losses[f"loss/{data_source_name}"] += mean_loss.item()
			final_losses[f"{loss_name}"] += mean_loss.item()
			final_losses[f"loss"] += mean_loss

		return final_losses

	def make_preview(self, batch, pred):
		"""
		Get visualizations of the predictions and the ground truth.
		"""
		data = batch # 由于特殊的no_collate_function，batch本身就是一个样本
		patches = unpack_dataset(data)

		H = data["H"]
		W = data["W"]
		T = len(patches)

		all_vis = torch.zeros((1, T, 1, H, 3*W), device=self.device, dtype=torch.uint8) # 3 channels: events, pred, gt

		gt_full = torch.zeros((1, H, W), device=self.device)
		pred_full = torch.zeros((1, H, W), device=self.device)
		for t in range(T):
			events = patches[t]["voxel"]
			gt_img = patches[t]["gt_img"]
			pred_img = pred[t]["pred"]
			ey = patches[t]["ey"]
			ex = patches[t]["ex"]
			eh = patches[t]["eh"]
			ew = patches[t]["ew"]

			event_vis = normalize_nobias(torch.sum(events, dim=0, keepdim=True))*255 # 1, eh, ew
			pred_vis = pred_img*255 # 1, eh, ew
			gt_vis = gt_img*255 # 1, eh, ew
			gt_full[:, ey:ey+eh, ex:ex+ew] = gt_vis
			pred_full[:, ey:ey+eh, ex:ex+ew] = pred_vis
			event_full = torch.zeros((1, H, W), device=self.device)
			event_full[:, ey:ey+eh, ex:ex+ew] = event_vis

			vis = torch.cat([event_full, pred_full, gt_full], dim=2) # 1, H, 3*W
			vis = vis.detach()
			vis = torch.clamp(vis, 0, 255)
			vis = vis.to(torch.uint8)
			all_vis[0, t] = vis

		# Convert to 3 channels
		all_vis = all_vis.repeat(1, 1, 3, 1, 1) # 3 channels
		return all_vis
