import torch
import numpy as np
import cv2
from typing import Union, List, Optional, Tuple, Dict, Any
from transformers.processing_utils import Unpack
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessorKwargs
from transformers import Qwen3VLProcessor, Qwen2VLImageProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.image_utils import ImageInput
from transformers.video_utils import VideoInput
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
from transformers.utils import logging
from PIL import Image
import os

from qwen_vl_utils import extract_vision_info, fetch_image


logger = logging.get_logger(__name__)

def visualize_patches(flatten_patches: torch.Tensor, out_pth: str):
	merge_size = 2
	channel = 3
	temporal_patch_size = 2
	patch_size = 16
	patches = flatten_patches.reshape(
		-1, merge_size, merge_size, channel, temporal_patch_size, patch_size, patch_size
	)
	patches = patches.permute(0, 4, 1, 5, 2, 6, 3) # (seq_len, temporal_patch_size, merge_size, patch_size, merge_size, patch_size, channel)
	patches = patches.reshape(-1, temporal_patch_size, merge_size * patch_size, merge_size * patch_size, channel).detach().numpy()
	patches = patches[:, 0, ..., ::-1]
	
	max_seq_len = 600
	seq_len, h, w, c = patches.shape
	if seq_len > max_seq_len:
		# If the sequence length exceeds the maximum, we will only visualize the first max_seq_len patches.
		patches = patches[:max_seq_len]
		seq_len = max_seq_len

	cols = 20
	rows = seq_len // cols + (1 if seq_len % cols > 0 else 0)
	width_per_patch = merge_size * patch_size + 2
	pw = merge_size * patch_size

	norm_patches = ((patches - patches.min()) / (patches.max() - patches.min()) * 255).astype(np.uint8)
	drawboard = np.zeros((rows * width_per_patch, cols * width_per_patch, c), dtype=np.uint8)
	for i in range(seq_len):
		row = i // cols
		col = i % cols
		drawboard[
			row * width_per_patch: row * width_per_patch + pw,
			col * width_per_patch: col * width_per_patch + pw
		] = norm_patches[i]
	
	cv2.imwrite(out_pth, drawboard)

def break_big_patches(big_patches):
	'''
	big_patches: List of items. Each item is:
	{
		"pixels": torch.Tensor of shape (H, W, 2). H and W are multiples of 32. 2 is temporal channel. Pixels are grayscale.
		"position": (y, x, h, w, begin_t, end_t).
	}
	Output is list of (32, 32, 2) patches with positions.
	'''
	small_patches = []
	patch_h, patch_w = 32, 32
	for item in big_patches:
		pixels = item["pixels"]
		y, x, h, w, begin_t, end_t = item["position"]
		H, W, T = pixels.shape
		assert H % patch_h == 0 and W % patch_w == 0, "H and W must be multiples of 32."
		assert T == 2, "Temporal size must be 2."

		for i in range(0, H, patch_h):
			for j in range(0, W, patch_w):
				patch = pixels[i:i+patch_h, j:j+patch_w, :]
				pos = (y + i, x + j, patch_h, patch_w, begin_t, end_t)
				small_patches.append({
					"pixels": patch,
					"position": pos
				})
	return small_patches

'''
Output format of patch-based video:
The file is a .npz file.

f["shapes"]: [N, 3]. f["shapes"][i] is the shape of the i-th patch. Should be (H, W, 2) (two sequential temporal frames).
f["flat_data"]: All the patches flattened and concatenated together.
f["indices"]: [N, 2]. f["indices"][i] is (begin_idx, end_idx) of the i-th patch, so that f["flat_data"][begin_idx:end_idx].reshape(f["shapes"][i]) is the i-th patch.
f["positions"]: [N, 6]. f["positions"][i] is (y, x, h, w, begin_t, end_t) of the i-th patch. The patch corresponds to img[y:y+h, x:x+w], is reconstructed from events between the time interval [begin_t, end_t], and corresponds to the moment end_t.
'''
def npz_to_large_patches_pair(npz_path: str) -> List[dict]:
	'''
	Returns:
	List of large patches. Each large patch is a dict:
	{
		"pixels": torch.Tensor of shape (H, W, 2). H and W are multiples of 32. 2 is temporal channel. Pixels are grayscale.
		"position": (y, x, h, w, begin_t, end_t).
	}
	'''
	npz_data = np.load(npz_path)
	shapes = npz_data['shapes'] # [N, 3]
	flat_data = npz_data['flat_data'] # all patches flattened and concatenated
	indices = npz_data['indices'] # [N, 2]
	positions = npz_data['positions'] # [N, 6]
	large_patches = []
	for i in range(len(shapes)):
		begin_idx, end_idx = indices[i]
		shape = shapes[i]
		patch = flat_data[begin_idx:end_idx].reshape(shape) # (H, W, 2)
		position = tuple(positions[i]) # (y, x, h, w, begin_t, end_t)
		large_patches.append({
			"pixels": torch.tensor(patch, dtype=torch.float32),
			"position": position
		})
	return large_patches

def npz_to_patches_pair(npz_path: str) -> List[dict]:
	npz_data = np.load(npz_path)
	shapes = npz_data['shapes']  # [N, 2]
	flat_data = npz_data['flat_data']  # all patches flattened and concatenated
	indices = npz_data['indices']  # [N, 2]
	positions = npz_data['positions']  # [N, 6]
	patches_pair = []

	# 存储历史patch的字典，key为(y, x)位置，value为patch数据和时间信息
	last_patches_dict = {}

	for i in range(len(shapes)):
		begin_idx, end_idx = indices[i]
		shape = shapes[i]
		patch = flat_data[begin_idx:end_idx].reshape(shape)  # (H, W)
		position = tuple(positions[i])  # (y, x, h, w, begin_t, end_t)
		y, x, h, w, begin_t, end_t = position
		y = int(y)
		x = int(x)
		h = int(h)
		w = int(w)
		for row in range(0, h, 32):
			for col in range(0, w, 32):
				small_patch_pos = (y + row, x + col)
				small_patch = patch[row:row+32, col:col+32]
				# 如果在历史记录中找到相同位置的patch，则配对
				if small_patch_pos in last_patches_dict:
					last_patch_info = last_patches_dict.pop(small_patch_pos)
					paired_patch = torch.stack([last_patch_info['patch'], torch.tensor(small_patch, dtype=torch.float32)], dim=-1)  # (32, 32, 2)
					paired_position = (small_patch_pos[0], small_patch_pos[1], 32, 32, last_patch_info['end_t']*2-end_t, end_t)
					patches_pair.append({
						"pixels": paired_patch,
						"position": paired_position
					})
					found_pair = True
				else:
					# 否则，存储当前patch以备后续配对
					last_patches_dict[small_patch_pos] = {
						'patch': torch.tensor(small_patch, dtype=torch.float32),
						'end_t': end_t
					}
	assert len(last_patches_dict) == 0, "Some patches did not find their pairs."
	return patches_pair


class EventVLProcessor(Qwen3VLProcessor):

	def __call__(
		self,
		images: ImageInput = None,
		text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
		patch_videos: List[List[dict]] = None, # List of videos reorganized in large_patch format
		tokens_per_frame: int = 128, # How many (32, 32, 2) areas (corresponding to a single token) are put into each attention window. Every tokens_per_frame patches, an <timestamp> text is inserted.
		saved_in_pair: bool = True,
		**kwargs: Unpack[Qwen3VLProcessorKwargs]
	) -> BatchFeature:
		"""
		Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
		and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
		the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
		Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

		Args:
			images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
				The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
				tensor. Both channels-first and channels-last formats are supported.
			text (`str`, `List[str]`, `List[List[str]]`):
				The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
				(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
				`is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
			patch_videos (`List[List[dict]]`, *optional*):
				The video or batch of videos to be prepared in large_patch format. Each video is a list of patches, and each patch is a dict:
				{
					"pixels": torch.Tensor of shape (H, W, 2). H and W are multiples of 32. 2 is temporal channel. Pixels are grayscale.
					"position": (y, x, h, w, begin_t, end_t).
				}
			patches_per_frame (`int`, *optional*, defaults to 128):
				How many (32, 32, 2) areas (corresponding to a single token) are put into each attention window. Every tokens_per_frame patches, an <timestamp> text is inserted.
			return_tensors (`str` or [`~utils.TensorType`], *optional*):
				If set, will return tensors of a particular framework. Acceptable values are:
				- `'tf'`: Return TensorFlow `tf.constant` objects.
				- `'pt'`: Return PyTorch `torch.Tensor` objects.
				- `'np'`: Return NumPy `np.ndarray` objects.
				- `'jax'`: Return JAX `jnp.ndarray` objects.

		Returns:
			[`BatchFeature`]: A [`BatchFeature`] with the following fields:

			- **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
			- **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
			  `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
			  `None`).
			- **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
			- **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
			- **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
			- **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
			- **second_per_grid_ts** -- List of video seconds per time grid. Returned when `videos` is not `None`.
		"""
		output_kwargs = self._merge_kwargs(
			Qwen3VLProcessorKwargs,
			tokenizer_init_kwargs=self.tokenizer.init_kwargs,
			**kwargs,
		)
		if images is not None:
			image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
			image_grid_thw = image_inputs["image_grid_thw"]
		else:
			image_inputs = {}
			image_grid_thw = None

		if patch_videos is not None:
			videos_inputs = self.large_patches_to_pixels(patch_videos, tokens_per_frame, saved_in_pair) 
			# output should be equivallent to original self.video_processor
			# video_inputs["pixel_values_videos"]: (total_num_tokens*4, 3*2*16*16) tensor
			# video_inputs["token_positions"]: (total_num_tokens, 7) tensor
			# video_inputs["tokens_per_video"]: [num_tokens for each video]
			videos_inputs["pixel_values_videos"] = self.rescale_and_normalize_pvv(videos_inputs["pixel_values_videos"])
			token_positions = videos_inputs["video_token_positions"]
			tokens_per_video = videos_inputs["tokens_per_video"]
		else:
			videos_inputs = {}
			token_positions = None
			tokens_per_video = None
		

		if not isinstance(text, list):
			text = [text]

		text = text.copy()  # below lines change text in-place
		if image_grid_thw is not None:
			merge_length = self.image_processor.merge_size**2
			index = 0
			for i in range(len(text)):
				while self.image_token in text[i]:
					num_image_tokens = image_grid_thw[index].prod() // merge_length
					text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
					index += 1
				text[i] = text[i].replace("<|placeholder|>", self.image_token)

		if token_positions is not None:
			merge_length = self.video_processor.merge_size**2 # 2*2 patches = 4
			index = 0 # 是第几个video
			for i in range(len(text)):
				while self.video_token in text[i]:

					video_placeholder = ""
					
					frame_seqlen = tokens_per_video[index]
					vid_begin_token_idx = torch.sum(tokens_per_video[:index]).item()
					for frame_begin_token_idx in range(0, frame_seqlen, tokens_per_frame):
						# TODO: assert token_positions is float
						curr_time = token_positions[vid_begin_token_idx + frame_begin_token_idx][5].item() # end_time
						#print("curr_time: ", curr_time)
						video_placeholder += f"<{curr_time:.2f} seconds>"
						num_tokens_in_this_window = min(tokens_per_frame, frame_seqlen - frame_begin_token_idx)
						video_placeholder += (
							self.vision_start_token + "<|placeholder|>" * num_tokens_in_this_window + self.vision_end_token
						)

					# 把单个的“这儿有一个video”替换成“这儿有N个video token”。
					if f"{self.vision_start_token}{self.video_token}{self.vision_end_token}" in text[i]:
						text[i] = text[i].replace(
							f"{self.vision_start_token}{self.video_token}{self.vision_end_token}", video_placeholder, 1
						)
					else:
						# vllm may input video token directly
						text[i] = text[i].replace(self.video_token, video_placeholder, 1)
					index += 1

				text[i] = text[i].replace("<|placeholder|>", self.video_token)

		return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
		return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
		text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
		self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

		if return_mm_token_type_ids:
			array_ids = np.array(text_inputs["input_ids"])
			mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
			mm_token_type_ids[array_ids == self.image_token_id] = 1
			text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

		return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

	def rescale_and_normalize_pvv(self, pixel_values_videos):
		# input shape: (batch_size * grid_t * grid_h//merge_size * grid_w//merge_size * merge_size * merge_size, channel * temporal_patch_size * patch_size * patch_size)
		arr = pixel_values_videos.reshape(-1, 3, 2, 16, 16)
		arr = arr.permute(0, 2, 1, 3, 4) # (N, T, C, H, W)
		arr = arr.reshape(-1, 3, 16, 16)
		arr = self.video_processor.rescale_and_normalize(
			images = arr,
			do_rescale = True,
			rescale_factor = 1 / 255.0,
			do_normalize = True,
			image_mean = (0.5, 0.5, 0.5),
			image_std = (0.5, 0.5, 0.5),
		) # params copied from video_processing_qwen3_vl.py
		return arr
	
	def large_patches_to_pixels(self, batch_large_patches: List[List[dict]], tokens_per_frame: int, saved_in_pair: bool = False) -> Dict[str, torch.Tensor]:
		'''
		batch_large_patches: List of videos. Each video is a list of large patches.
		Each large patch is a dict:
		{
			"pixels": torch.Tensor of shape (H, W, 2). H and W are multiples of 32. 2 is temporal channel. Pixels are grayscale.
			"position": (y, x, h, w, begin_t, end_t).
		}
		Returns:
		{
			"pixel_values_videos": List of torch.Tensor of shape (num_tokens*4, 3*2*16*16) for each video.
			"video_token_positions": List of torch.Tensor of shape (num_tokens, 7) for each video. Each position is [Y_top, X_left, H, W, T_begin, T_end, frame_idx].
			"tokens_per_video": List of int, number of tokens for each video.
		}
		First, break all large patches into (32, 32, 2) token patches. (Define each (32, 32, 2) patch as a token patch.)
		Then, sort them according to time. Split them into "frames" according to tokens_per_frame.
		Within each frame, sort the token patches according to (y, x).
		Finally, convert the (32, 32, 2) grayscale patches into (16, 16, 2, 3) RGB patches by duplicating channels and splitting each one into 4 small patches.
		'''
		pixel_values_videos = []
		token_positions = []
		tokens_per_video = []
		video_cnt = len(batch_large_patches)
		
		for video_idx in range(video_cnt):
			large_patches = batch_large_patches[video_idx]
			if (saved_in_pair):
				token_patches = break_big_patches(large_patches) # List of (32, 32, 2) token patches with positions.
			else:
				token_patches =large_patches

			# Sort token patches according to time.
			token_patches.sort(key=lambda x: x["position"][5]) # end_t

			# Split into frames according to tokens_per_frame
			for i in range(0, len(token_patches), tokens_per_frame):
				# sort within each frame according to (y, x)
				frame_patches = token_patches[i:i+tokens_per_frame]
				frame_patches.sort(key=lambda x: (x["position"][0], x["position"][1])) # (y, x)
				token_patches[i:i+tokens_per_frame] = frame_patches

			vid_pixel_values = []
			vid_token_positions = []
			# Convert each (32, 32, 2) grayscale patch into 4 (16, 16, 2, 3) RGB patches.
			for patch_idx, patch in enumerate(token_patches):
				rgb_patch = patch["pixels"].unsqueeze(-1).repeat(1, 1, 1, 3) # (H, W, T, C) = (32, 32, 2, 3)
				rgb_patch = rgb_patch.permute(2, 3, 0, 1) # (T, C, H, W)
				rgb_patch = rgb_patch.view(
					1, # 0 batch_size=1
					1, # 1 grid_t=1
					2, # 2 temporal_patch_size=2
					3, # 3 channel
					1, # 4 grid_h=2 // merge_size=2
					2, # 5 merge_size
					16, # 6 patch_size
					1, # 7 grid_w=2 // merge_size=2,
					2, # 8 merge_size
					16  # 9 patch_size
				)
				rgb_patch = rgb_patch.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
				# shape of rgb_patch now: (batch_size, grid_t, grid_h//merge_size, grid_w//merge_size, merge_size, merge_size, channel, temporal_patch_size, patch_size, patch_size)
				rgb_patch = rgb_patch.reshape(
					4, # batch_size*grid_t*grid_h*grid_w
					3 * 2 * 16 * 16 # channel * temporal_patch_size * patch_size * patch_size
				) 
				vid_pixel_values.append(rgb_patch)
				frame_idx = patch_idx // tokens_per_frame
				position = patch["position"] + (frame_idx,) # (y, x, h, w, begin_t, end_t, frame_idx)
				vid_token_positions.append(torch.tensor(position, dtype=torch.float64))
			
			pixel_values_videos.append(torch.cat(vid_pixel_values, dim=0)) # (num_tokens*4, 3*2*16*16)
			token_positions.append(torch.stack(vid_token_positions, dim=0)) # (num_tokens, 7)
			tokens_per_video.append(len(token_patches))

		return {
			"pixel_values_videos": torch.cat(pixel_values_videos, dim=0),
			"video_token_positions": torch.cat(token_positions, dim=0),
			"tokens_per_video": torch.tensor(tokens_per_video)
		}
	

def process_vision_info_patched(
	conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
	image_patch_size: int = 16,saved_in_pair: bool = False
) -> Tuple[Optional[List[Image.Image]], Optional[List[List[dict]]]]:
	vision_infos = extract_vision_info(conversations)
	## Read images or videos
	image_inputs = []
	video_inputs = []
	for vision_info in vision_infos:
		if "image" in vision_info or "image_url" in vision_info:
			image_inputs.append(fetch_image(vision_info, image_patch_size=image_patch_size))
		elif "video" in vision_info:
			# Only support .npz files
			vid_path = vision_info.get("video", None)
			assert vid_path is not None, "video path is None."
			assert vid_path.endswith('.npz'), "Only .npz video files are supported in this patched function."
			if saved_in_pair:
				patch_video = npz_to_large_patches_pair(vid_path)
			else:
				patch_video = npz_to_patches_pair(vid_path)
			video_inputs.append(patch_video)
		else:
			raise ValueError("image, image_url or video should in content.")
	if len(image_inputs) == 0:
		image_inputs = None
	if len(video_inputs) == 0:
		video_inputs = None

	return image_inputs, video_inputs