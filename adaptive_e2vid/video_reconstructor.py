import sys
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)
if PACKAGE_DIR not in sys.path:
	sys.path.insert(0, PACKAGE_DIR)
if PROJECT_DIR not in sys.path:
	sys.path.insert(0, PROJECT_DIR)

import numpy as np
import torch
import yaml
from adaptive_e2vid.model.model import E2VIDRecurrent, FlowNet
from adaptive_e2vid.model.model import E2VIDSparse2
import threading
import h5py
import cv2
import tempfile
import subprocess
import shutil
import argparse

from adaptive_e2vid.data.sparse_testh5 import SparseTestH5Dataset

def convert_to_compiled(state_dict, local_rank, use_compile=False):
	new_dict = {}
	for k, v in state_dict.items():
		parts = k.split(".")
		# First pop out "_orig_mod" and "module"
		if parts[0] == "_orig_mod":
			parts.pop(0)
		if parts[0] == "module":
			parts.pop(0)

		# Then, add the required "_orig_mod" and "module" back
		if local_rank is not None:
			# Should be DDP
			parts.insert(0, "module")
		# Use torch.compile
		if use_compile:
			parts.insert(0, "_orig_mod")

		new_k = ".".join(parts)
		new_dict[new_k] = v

	return new_dict

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

def h5_to_voxel(h5_path, bin_num):
	with h5py.File(h5_path, 'r') as f:                
		ts = f["events/ts"][:]
		xs = f["events/xs"][:]
		ys = f["events/ys"][:]
		ps = f["events/ps"][:].astype(np.int8)
		ps = np.where(ps > 0, 1, -1)  # Convert polarities to 1 and -1
		
		try:
			H , W = f.attrs['sensor_resolution']
		except:
			H = np.max(ys) + 1
			W = np.max(xs) + 1

		voxel = np.zeros((bin_num, H, W))
		t_per_bin = (ts[-1] - ts[0] + 1e-6) / bin_num # Avoid division by zero
		# Print max and min and [-1] and [0] of ts
		# print(f"Event timestamps: min {np.min(ts)}, max {np.max(ts)}, duration {ts[-1]-ts[0]}, begin {ts[0]}, end {ts[-1]}")
		# print(f"Event spatial resolution: H {H}, W {W}")
		#print(f"t_per_bin: {t_per_bin}, bin_num: {bin_num}")

		# Discard all events with ts > ts[-1] or ts < ts[0]
		if np.max(ts) > ts[-1] or np.min(ts) < ts[0]:
			valid_mask = (ts >= ts[0]) & (ts <= ts[-1])
			print(f"Warning: Discarding {np.sum(~valid_mask)} events out of bounds [{ts[0]}, {ts[-1]}]")
			ts = ts[valid_mask]
			xs = xs[valid_mask]
			ys = ys[valid_mask]
			ps = ps[valid_mask]

		bin_idx = ((ts - ts[0]) / t_per_bin).astype(np.int32)
		np.add.at(voxel, (bin_idx, ys, xs), ps)
	return voxel

class VideoReconstructor:
	def __init__(self):
		script_dir = os.path.dirname(os.path.abspath(__file__))
		
		self.e2vid_model = E2VIDRecurrent(
			unet_kwargs={
				"num_bins": 5,
				"skip_type": "sum",
				"recurrent_block_type": "convlstm",
				"num_encoders": 3,
				"base_num_channels": 32,
				"num_residual_blocks": 2,
				"use_upsample_conv": True,
				"final_activation": "",
				"norm": "none"
			}
		)
		ckpt_path = os.path.join(script_dir, "checkpoints/v2v_e2vid_10k/epoch_0077.pth")
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.e2vid_model = self.e2vid_model.to(self.device)		
		
		state_dict = torch.load(ckpt_path,weights_only = False)["state_dict"]
		state_dict = convert_to_compiled(state_dict=state_dict, local_rank=None, use_compile=False)
		self.e2vid_model.load_state_dict(state_dict, strict=False)
		self.e2vid_model.eval()
		# set to no_grad
		torch.set_grad_enabled(False)
		
		# 添加线程锁
		self._reconstruct_lock = threading.Lock()
	
	def reconstruct(self, voxel: np.ndarray) -> np.ndarray:
		#print("========Voxel shape: ", voxel.shape)
		"""
		Args:
			voxel: (T, 5, H, W) numpy array, voxel grid of events
		Returns:
			frames: (T, H, W) numpy array, reconstructed frames
		"""
		# 使用锁确保同一时间只有一个reconstruct操作在进行
		with self._reconstruct_lock:
			assert len(voxel.shape) == 4 and voxel.shape[1] == 5, "voxel should be of shape (T, 5, H, W)"
			T, _, H, W = voxel.shape
			PAD = 16
			padded_h = int(np.ceil(H / PAD) * PAD)
			padded_w = int(np.ceil(W / PAD) * PAD)
			padded_voxel = np.zeros((T, 5, padded_h, padded_w), dtype=np.int8)
			padded_voxel[:, :, :H, :W] = voxel
			output_vid = np.zeros((T, H, W), dtype=np.uint8)
			#print("Will begin reconstruction, padded_voxel mem usage: ", padded_voxel.nbytes / (1024*1024*1024), "GB", "output_vid mem usage: ", output_vid.nbytes / (1024*1024*1024), "GB")

			self.e2vid_model.reset_states()
			for t in range(T):
				# Frame by frame, prevent high GPU mem cost by moving all to GPU at once
				v = torch.from_numpy(padded_voxel[t]).float().to(self.device)
				pred = self.e2vid_model(v.unsqueeze(0))  # (1, 1, H, W)
				pred = pred["image"].detach().squeeze().cpu().numpy()[:H, :W] * 255
				pred = np.clip(pred, 0, 255).astype(np.uint8)
				output_vid[t] = pred

			return output_vid

	def h5_to_video(self, h5_path, video_path, FPS=24):
		"""Convert .h5 event data to video file using FFmpeg"""
		with h5py.File(h5_path, 'r') as f:
			H, W = f.attrs["sensor_resolution"]   
			ts_dtype = f["events/ts"].dtype
			t_0, t_1 = f["events/ts"][0], f["events/ts"][-1]
			if ts_dtype not in [np.int64, np.uint64, np.int32, np.uint32]:
				t_0, t_1 = int(t_0 * 1e6), int(t_1 * 1e6)
			total_frame_cnt = int((t_1 - t_0) * FPS / 1e6) + 1  # Convert microseconds to seconds
			e2vid_voxels = h5_to_voxel(h5_path, 5 * total_frame_cnt).reshape((total_frame_cnt, 5, H, W))
			recon = self.reconstruct(e2vid_voxels)
			success = np_to_video(recon, video_path, fps=FPS)
			assert success
		return True


class VideoReconstructor_Adaptive:
	def __init__(self):
		script_dir = os.path.dirname(os.path.abspath(__file__))

		self.e2vid_model = E2VIDSparse2(
			unet_kwargs={
				"num_bins": 5,
				"skip_type": "sum",
				"recurrent_block_type": "convlstm",
				"num_encoders": 3,
				"base_num_channels": 32,
				"num_residual_blocks": 2,
				"use_upsample_conv": True,
				"final_activation": "",
				"norm": "none"
			}
		)
		cpkt_path = os.path.join(script_dir, "checkpoints/adaptive_e2vid/epoch_0059.pth")
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.e2vid_model = self.e2vid_model.to(self.device)

		state_dict = torch.load(cpkt_path,weights_only = False)["state_dict"]
		state_dict = convert_to_compiled(state_dict=state_dict, local_rank=None, use_compile=False)
		self.e2vid_model.load_state_dict(state_dict, strict=False)
		self.e2vid_model.eval()
		# set to no_grad
		torch.set_grad_enabled(False)

		self._lock = threading.Lock()

	def _save_npz(self, patches, pred_list, npz_path):
		shapes = []
		flat_data = []
		indices = []
		positions = []
		current_index = 0

		T = len(patches)
		for i in range(T):
			patch = patches[i]
			pred_img = pred_list[i]["pred"] * 255
			p = torch.clamp(pred_img, 0, 255).squeeze().detach().cpu().numpy().astype(np.uint8)

			shapes.append([p.shape[0], p.shape[1]])
			flat_patch = p.flatten()
			flat_data.append(flat_patch)
			begin_idx = current_index
			end_idx = current_index + len(flat_patch)
			indices.append([begin_idx, end_idx])
			current_index = end_idx

			positions.append([
				patch["ey"],
				patch["ex"],
				patch["eh"],
				patch["ew"],
				patch["begin_t"],
				patch["end_t"],
			])

		if not flat_data:
			print(f"No data to save for {npz_path}")
			return False

		flat_data_concatenated = np.concatenate(flat_data)
		shapes_np = np.array(shapes)
		indices_np = np.array(indices)
		positions_np = np.array(positions)

		dirname = os.path.dirname(npz_path)
		if dirname:
			os.makedirs(dirname, exist_ok=True)
		np.savez(
			npz_path,
			shapes=shapes_np,
			flat_data=flat_data_concatenated,
			indices=indices_np,
			positions=positions_np,
		)
		return True

	def _move_batch_to_device(self, batch):
		batch_on_device = {}
		for key, value in batch.items():
			if isinstance(value, torch.Tensor):
				batch_on_device[key] = value.to(self.device)
			else:
				batch_on_device[key] = value
		return batch_on_device
	
	def h5_to_video(self, h5_path, npz_path):
		with self._lock:
			dataset = SparseTestH5Dataset(h5_path, configs={
				"num_bins": 5,
				"trigger_type": "merge",
				"bin_type": "events",
				"patch_dim": 2,
				"patch_per_time": 2,
				"trigger_param_gen_name": "fixed_threshold",
			})
			batch = dataset[0]
			data = self._move_batch_to_device(batch)

			H = data["H"]
			W = data["W"]
			T = data["begin_idx_arr"].shape[0]
			patches = unpack_dataset(data)
			device = data["voxel_flat"].device

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
			
			success = self._save_npz(patches, pred_patches, npz_path)
			return success
		

def np_to_video(imgs, output_path, fps=24):
	"""Convert numpy array of images to video file using FFmpeg"""

	# Use FFmpeg to create web-compatible video        
	# Create temporary directory for frames
	temp_dir = tempfile.mkdtemp()
	frames_dir = os.path.join(temp_dir, 'frames')
	os.makedirs(frames_dir)
	
	success = False
	try:
		print(f"Creating {len(imgs)} frames in: {frames_dir}")
		
		# Save frames as PNG images
		for i in range(len(imgs)):
			frame = imgs[i]
			if len(frame.shape) == 2:  # Grayscale to BGR
				frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
			frame_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
			cv2.imwrite(frame_path, frame)
		
		# Use FFmpeg to create H.264 video with web-compatible settings
		ffmpeg_cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-framerate', str(fps),
            '-i', os.path.join(frames_dir, 'frame_%06d.png'),
            # Ensure even dimensions for yuv420p/libx264
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            '-c:v', 'libx264',  # H.264 codec
            '-pix_fmt', 'yuv420p',  # Browser-compatible pixel format
            '-crf', '23',  # Quality setting (lower = better quality)
            '-preset', 'medium',  # Encoding speed vs compression
            '-movflags', '+faststart',  # Optimize for web streaming
            '-r', str(fps),  # Output framerate
            output_path
        ]
		
		print(f"Running FFmpeg: {' '.join(ffmpeg_cmd)}")
		result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=120)
		
		if result.returncode == 0:
			file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
			print(f"FFmpeg successful! Video created: {output_path}, size: {file_size} bytes")
			success = file_size > 0
		else:
			print(f"FFmpeg failed with return code {result.returncode}")
			print(f"FFmpeg stderr: {result.stderr}")
			print(f"FFmpeg stdout: {result.stdout}")
			success = False
		
	except subprocess.TimeoutExpired:
		print("FFmpeg process timed out")
		success = False
	except FileNotFoundError:
		print("FFmpeg not found. Please install FFmpeg: sudo apt install ffmpeg")
		success = False
	except Exception as e:
		print(f"Error running FFmpeg: {e}")
		success = False
	finally:
		# Clean up temporary frames directory
		try:
			shutil.rmtree(temp_dir)
			print(f"Cleaned up temporary directory: {temp_dir}")
		except Exception as e:
			print(f"Error cleaning up temp directory: {e}")
	
	return success
		

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Reconstruct a video from an HDF5 event file using a pre-trained E2VID model."
	)
	parser.add_argument("h5_path", help="Path to the input HDF5 event file.")
	parser.add_argument("video_path", help="Path to the output video file.")
	parser.add_argument(
		"--fps",
		type=int,
		default=30,
		help="Frames per second for the output video. Default: 30",
	)
	args = parser.parse_args()

	h5_path = args.h5_path
	video_path = args.video_path
	target_fps = args.fps

	# Check if input file exists
	if not os.path.exists(h5_path):
		print(f"Error: Input file not found at {h5_path}")
		sys.exit(1)

	# Ensure output directory exists
	output_dir = os.path.dirname(video_path)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)

	reconstructor = VideoReconstructor()

	# Determine number of frames based on duration and target FPS
	with h5py.File(h5_path, 'r') as f:
		duration_s = (f["events/ts"][-1] - f["events/ts"][0])
	# If > 100000, assume timestamps are in microseconds
	if duration_s > 100000:
		duration_s /= 1e6
	
	num_frames = int(duration_s * target_fps)
	print(f"Event stream duration: {duration_s:.2f}s. Reconstructing {num_frames} frames for a target FPS of {target_fps}.")

	# Convert h5 to voxel grid
	voxel = h5_to_voxel(h5_path, num_frames*5)
	_, H, W = voxel.shape
	
	# Reshape voxel grid for the model (T, 5, H, W)
	# The model expects event bins of size 5.
	num_model_frames = len(voxel) // 5
	voxel = voxel[:num_model_frames * 5] # Discard trailing bins
	voxel = voxel.reshape((num_model_frames, 5, H, W))

	if voxel.shape[0] == 0:
		print("Error: Not enough event data to create even one frame.")
		sys.exit(1)

	# Reconstruct frames from voxel grid
	print("Reconstructing video...")
	pred_frames = reconstructor.reconstruct(voxel)
	print(f"Reconstruction complete. Shape: {pred_frames.shape}")

	# Save frames as a video file
	print(f"Saving video to {video_path}...")
	success = np_to_video(pred_frames, video_path, fps=target_fps)

	if success:
		print("Video saved successfully.")
	else:
		print("Failed to save video.")