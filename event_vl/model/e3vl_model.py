from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLModel, Qwen3VLTextModel, Qwen3VLVisionModel
from typing import Union, List, Optional, Tuple
from transformers.processing_utils import Unpack
import torch
import time
from transformers.utils import is_torchdynamo_compiling
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast, Qwen3VLModelOutputWithPast
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.utils import TransformersKwargs
from transformers.cache_utils import Cache
import numpy as np

def tensor_to_txt(ts: torch.Tensor, path: str):
	# Write out all elements of the tensor into a txt file in a readable format.
	np.savetxt(path, ts.detach().cpu().numpy(), fmt='%.2f')

class EventVLVisionModel(Qwen3VLVisionModel):

	def __init__(self, config: Qwen3VLVisionConfig):
		super().__init__(config)
		self.head_dim = config.hidden_size // config.num_heads

	def forward(
			self, 
			hidden_states: torch.Tensor, 
			grid_thw: Optional[torch.Tensor] = None,
			video_token_positions: Optional[torch.Tensor] = None,
			tokens_per_video: Optional[torch.Tensor] = None,
		) -> torch.Tensor:
		if grid_thw is not None:
			return self.forward_image(hidden_states, grid_thw=grid_thw)
		elif video_token_positions is not None:
			return self.forward_video(hidden_states, video_token_positions=video_token_positions, tokens_per_video=tokens_per_video)
		else:
			raise ValueError("Either grid_thw or video_token_positions must be provided.")

	def rot_pos_emb_vtp(self, video_token_positions_split: List[torch.Tensor]) -> torch.Tensor:
		merge_size = self.spatial_merge_size
		total_tokens = sum([pos.shape[0] for pos in video_token_positions_split])

		block_rows = torch.cat([tok_pos[:, 0] for tok_pos in video_token_positions_split], dim=0) // 32  # (total_tokens,)
		block_cols = torch.cat([tok_pos[:, 1] for tok_pos in video_token_positions_split], dim=0) // 32 # (total_tokens,)
		intra_row = torch.arange(merge_size, device=block_rows.device)  # (merge_size,)
		intra_col = torch.arange(merge_size, device=block_cols.device)  # (merge_size,)
		row_idx = block_rows[:, None, None] * merge_size + intra_row[None, :, None]  # (total_tokens, merge_size)
		col_idx = block_cols[:, None, None] * merge_size + intra_col[None, None, :]  # (total_tokens, merge_size)

		row_idx = row_idx.expand(total_tokens, merge_size, merge_size).reshape(-1)
		col_idx = col_idx.expand(total_tokens, merge_size, merge_size).reshape(-1)

		pos_ids = torch.stack((row_idx, col_idx), dim=-1)  # (total_tokens * merge_size * merge_size, 2)

		max_hw = max(row_idx.max().item(), col_idx.max().item()) + 1
		freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
		# TODO: 改成现场算rotary pos emb，适应于float的空间坐标（非32整数倍坐标）

		embeddings = freq_table[pos_ids.long()]  # lookup rotary embeddings
		embeddings = embeddings.flatten(1)
		return embeddings

	def fast_pos_embed_interpolate_vtp(self, video_token_positions_split: List[torch.Tensor]) -> torch.Tensor:

		idx_list = [[] for _ in range(4)]
		weight_list = [[] for _ in range(4)]

		vid_max_hw = []

		for token_positions in video_token_positions_split:
			h = int(token_positions[:, 0].max().item() // 32 + 1)	* 2
			w = int(token_positions[:, 1].max().item() // 32 + 1)	* 2
			vid_max_hw.append((h, w))

			h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h) # (h,)
			w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w) # (w,)

			h_idxs_floor = h_idxs.int()
			w_idxs_floor = w_idxs.int()
			h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
			w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

			dh = h_idxs - h_idxs_floor
			dw = w_idxs - w_idxs_floor

			base_h = h_idxs_floor * self.num_grid_per_side
			base_h_ceil = h_idxs_ceil * self.num_grid_per_side

			# .T is transpose, for broadcasting
			# base_h[None]: (1, h); base_h[None].T: (h, 1)
			indices = [
				(base_h[None].T + w_idxs_floor[None]).flatten(), # (h, w) -> (h*w,)
				(base_h[None].T + w_idxs_ceil[None]).flatten(),
				(base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
				(base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
			]

			weights = [
				((1 - dh)[None].T * (1 - dw)[None]).flatten(),
				((1 - dh)[None].T * dw[None]).flatten(),
				(dh[None].T * (1 - dw)[None]).flatten(),
				(dh[None].T * dw[None]).flatten(),
			]

			for i in range(4):
				idx_list[i].extend(indices[i].tolist())
				weight_list[i].extend(weights[i].tolist())

		idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
		weight_tensor = torch.tensor(
			weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
		)
		pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
		patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

		patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in vid_max_hw])

		patch_pos_embeds_permute = []
		merge_size = self.config.spatial_merge_size
		
		for video_idx, (token_positions, pos_embed) in enumerate(zip(video_token_positions_split, patch_pos_embeds)):

			block_rows = token_positions[:, 0] // 32
			block_cols = token_positions[:, 1] // 32
			intra_row = torch.arange(merge_size, device=block_rows.device)
			intra_col = torch.arange(merge_size, device=block_cols.device)
			row_idx = block_rows[:, None, None] * merge_size + intra_row[None, :, None]
			col_idx = block_cols[:, None, None] * merge_size + intra_col[None, None, :]
			row_idx = row_idx.expand(token_positions.shape[0], merge_size, merge_size).reshape(-1)
			col_idx = col_idx.expand(token_positions.shape[0], merge_size, merge_size).reshape(-1)
			idx_in_embed = row_idx * (vid_max_hw[video_idx][1]) + col_idx

			embed = pos_embed[idx_in_embed.long()]
			patch_pos_embeds_permute.append(embed)

		patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
		return patch_pos_embeds

	def forward_image(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
		return super().forward(hidden_states=hidden_states, grid_thw=grid_thw, **kwargs)
	
	def forward_video(self, hidden_states: torch.Tensor, video_token_positions: torch.Tensor, tokens_per_video: torch.Tensor, **kwargs) -> torch.Tensor:
		"""
		Args:
			hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
				The final hidden states of the model.
			grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
				The temporal, height and width of feature shape of each image in LLM.

		Returns:
			`torch.Tensor`: hidden_states.
		"""
		hidden_states = self.patch_embed(hidden_states)
		video_token_positions_split = torch.split(
			video_token_positions, tokens_per_video.tolist(), dim=0
		)

		pos_embeds = self.fast_pos_embed_interpolate_vtp(video_token_positions_split)
		hidden_states = hidden_states + pos_embeds

		rotary_pos_emb = self.rot_pos_emb_vtp(video_token_positions_split)

		seq_len, _ = hidden_states.size()
		hidden_states = hidden_states.reshape(seq_len, -1)
		rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
		emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
		position_embeddings = (emb.cos(), emb.sin())

		vid_seqlens = []
		for video_idx in range(len(video_token_positions_split)):
			# Take out the frame_idx array, which is the frame_idx of each token. It is like [0, 0, 0, 1, 1, 3, 3, 3, 3]
			# Count the sequence length of each frame. Extend vid_seqlens with [3, 2, 4].
			token_positions = video_token_positions_split[video_idx]
			frame_indices = token_positions[:, 6]
			_, counts = torch.unique_consecutive(frame_indices, return_counts=True)
			vid_seqlens.extend(counts.tolist())
		cu_seqlens = torch.tensor(vid_seqlens).cumsum(dim=0)
		cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
		cu_seqlens = cu_seqlens * 4 # from tokens to patches

		deepstack_feature_lists = []
		for layer_num, blk in enumerate(self.blocks):
			hidden_states = blk(
				hidden_states,
				cu_seqlens=cu_seqlens,
				position_embeddings=position_embeddings,
				**kwargs,
			)
			if layer_num in self.deepstack_visual_indexes:
				deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
					hidden_states
				)
				deepstack_feature_lists.append(deepstack_feature)

		hidden_states = self.merger(hidden_states)

		return hidden_states, deepstack_feature_lists



class EventVLModel(Qwen3VLModel):
	
	def __init__(self, config, do_post_init=True):
		# Skip Qwen3VLModel.__init__ and call Qwen3VLPreTrainedModel.__init__ directly
		# to avoid duplicate initialization of visual and language_model
		super(Qwen3VLModel, self).__init__(config)
		self.visual = EventVLVisionModel._from_config(config.vision_config)
		self.language_model = Qwen3VLTextModel._from_config(config.text_config)
		self.rope_deltas = None  # cache rope_deltas here

		if do_post_init:
			# Initialize weights and apply final processing
			self.post_init()

	def get_video_features(
		self, pixel_values_videos: torch.FloatTensor, video_token_positions: torch.FloatTensor, tokens_per_video: torch.Tensor
	):
		"""
		Encodes videos into continuous embeddings that can be forwarded to the language model.

		Args:
			pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
				The tensors corresponding to the input videos.
			video_token_positions (List of `torch.Tensor` with shape `(sequence_length, 7)`):
	
		"""
		pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
		video_embeds = self.visual(pixel_values_videos, video_token_positions=video_token_positions, tokens_per_video=tokens_per_video)
		
		return video_embeds

	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Cache] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		pixel_values: Optional[torch.Tensor] = None,
		pixel_values_videos: Optional[torch.FloatTensor] = None,
		image_grid_thw: Optional[torch.LongTensor] = None,
		video_token_positions: Optional[torch.Tensor] = None,
		tokens_per_video: Optional[torch.Tensor] = None,
		cache_position: Optional[torch.LongTensor] = None,
		**kwargs: Unpack[TransformersKwargs],
	) -> Union[tuple, Qwen3VLModelOutputWithPast]:
		r"""
		image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
			The temporal, height and width of feature shape of each image in LLM.
		video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
			The temporal, height and width of feature shape of each video in LLM.
		"""
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

		if inputs_embeds is None:
			inputs_embeds = self.get_input_embeddings()(input_ids)

		image_mask = None
		video_mask = None

		if pixel_values is not None:
			image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
			image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
			image_mask, _ = self.get_placeholder_mask(
				input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
			)
			inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

		if pixel_values_videos is not None:
			video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_token_positions, tokens_per_video)
			video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
			_, video_mask = self.get_placeholder_mask(
				input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
			)
			inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

		visual_pos_masks = None
		deepstack_visual_embeds = None
		if image_mask is not None and video_mask is not None:
			# aggregate visual_pos_masks and deepstack_visual_embeds
			image_mask = image_mask[..., 0]
			video_mask = video_mask[..., 0]
			visual_pos_masks = image_mask | video_mask
			deepstack_visual_embeds = []
			image_mask_joint = image_mask[visual_pos_masks]
			video_mask_joint = video_mask[visual_pos_masks]
			for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
				embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
				embed_joint[image_mask_joint, :] = img_embed
				embed_joint[video_mask_joint, :] = vid_embed
				deepstack_visual_embeds.append(embed_joint)
		elif image_mask is not None:
			image_mask = image_mask[..., 0]
			visual_pos_masks = image_mask
			deepstack_visual_embeds = deepstack_image_embeds
		elif video_mask is not None:
			video_mask = video_mask[..., 0]
			visual_pos_masks = video_mask
			deepstack_visual_embeds = deepstack_video_embeds

		if position_ids is None:
			attention_mask_tensor = (
				attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
			)
			if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
				attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
				# Only apply conversion for floating point tensors (inverted masks)
				if attention_mask_tensor.dtype.is_floating_point:
					attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
					attention_mask_tensor = (1.0 - attention_mask_tensor).int()

			# Calculate RoPE index once per generation in the pre-fill stage only.
			# When compiling, we can't check tensor values thus we check only input length
			# It is safe to assume that `length!=1` means we're in pre-fill because compiled
			# models currently cannot do asssisted decoding
			prefill_compiled_stage = is_torchdynamo_compiling() and (
				(input_ids is not None and input_ids.shape[1] != 1)
				or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
			)
			prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
				(cache_position is not None and cache_position[0] == 0)
				or (past_key_values is None or past_key_values.get_seq_length() == 0)
			)
			if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
				position_ids, rope_deltas = self.get_rope_index(
					input_ids,
					image_grid_thw,
					video_token_positions,
					tokens_per_video,
					attention_mask=attention_mask_tensor,
				)
				self.rope_deltas = rope_deltas
			# then use the prev pre-calculated rope-deltas to get the correct position ids
			else:
				batch_size, seq_length, _ = inputs_embeds.shape
				delta = (
					(cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
					if cache_position is not None
					else 0
				)
				position_ids = torch.arange(seq_length, device=inputs_embeds.device)
				position_ids = position_ids.view(1, -1).expand(batch_size, -1)
				if cache_position is not None:  # otherwise `deltas` is an int `0`
					delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
				position_ids = position_ids.add(delta)
				position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

		outputs = self.language_model(
			input_ids=None,
			position_ids=position_ids,
			attention_mask=attention_mask,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			cache_position=cache_position,
			visual_pos_masks=visual_pos_masks,
			deepstack_visual_embeds=deepstack_visual_embeds,
			**kwargs,
		)

		return Qwen3VLModelOutputWithPast(
			last_hidden_state=outputs.last_hidden_state,
			past_key_values=outputs.past_key_values,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			rope_deltas=self.rope_deltas,
		)

	def get_rope_index(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		image_grid_thw: Optional[torch.LongTensor] = None,
		video_token_positions: Optional[torch.Tensor] = None,
		tokens_per_video: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

		if video_token_positions is not None:
			# split concated video_token_positions according to tokens_per_video
			video_token_positions_split = torch.split(
				video_token_positions, tokens_per_video.tolist(), dim=0
			)
			frame_seqlens = []
			for video_idx in range(len(video_token_positions_split)):
				# Take out the frame_idx array, which is the frame_idx of each token. It is like [0, 0, 0, 1, 1, 3, 3, 3, 3]
				# Count the sequence length of each frame. Extend vid_seqlens with [3, 2, 4].
				token_positions = video_token_positions_split[video_idx]
				frame_indices = token_positions[:, 6]
				_, counts = torch.unique_consecutive(frame_indices, return_counts=True)
				frame_seqlens.extend(counts.tolist())
			video_token_positions_split_by_frame = torch.split(
				video_token_positions, frame_seqlens, dim=0
			)

		spatial_merge_size = self.config.vision_config.spatial_merge_size
		image_token_id = self.config.image_token_id
		video_token_id = self.config.video_token_id
		vision_start_token_id = self.config.vision_start_token_id
		mrope_position_deltas = []
		if input_ids is not None and (image_grid_thw is not None or video_token_positions is not None):
			total_input_ids = input_ids
			if attention_mask is None:
				attention_mask = torch.ones_like(total_input_ids)
			position_ids = torch.ones(
				3,
				input_ids.shape[0], # batch size
				input_ids.shape[1], # sequence length
				dtype=input_ids.dtype,
				device=input_ids.device,
			)
			image_index, video_index = 0, 0
			attention_mask = attention_mask.to(total_input_ids.device)
			for i, input_ids in enumerate(total_input_ids):
				input_ids = input_ids[attention_mask[i] == 1]
				image_nums, video_nums = 0, 0
				vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1) # (num,) array of indices; corresponding to starts of vision sections
				vision_tokens = input_ids[vision_start_indices + 1] # 每个vision section取一个开头，判断是视频还是图像
				image_nums = (vision_tokens == image_token_id).sum()
				video_nums = (vision_tokens == video_token_id).sum()
				input_tokens = input_ids.tolist()
				llm_pos_ids_list: list = []
				st = 0
				remain_images, remain_videos = image_nums, video_nums
				for _ in range(image_nums + video_nums):
					# ed_x: find end of text section, so input_ids[st:ed_x] is all text. The <|vision_start|> is counted as text.
					if image_token_id in input_tokens and remain_images > 0:
						ed_image = input_tokens.index(image_token_id, st)
					else:
						ed_image = len(input_tokens) + 1
					if video_token_id in input_tokens and remain_videos > 0:
						ed_video = input_tokens.index(video_token_id, st)
					else:
						ed_video = len(input_tokens) + 1
					if ed_image < ed_video:
						# Will be processing image next
						t, h, w = (
							image_grid_thw[image_index][0],
							image_grid_thw[image_index][1],
							image_grid_thw[image_index][2],
						)
						image_index += 1
						remain_images -= 1
						ed = ed_image

						llm_grid_t, llm_grid_h, llm_grid_w = (
							t.item(), # 1
							h.item() // spatial_merge_size,
							w.item() // spatial_merge_size,
						) # 各个维度上的token数量

						# 视频前面一段text的长度
						text_len = ed - st
						# rope_index of text section: [(st_idx, st_idx, st_idx), (st_idx+1, st_idx+1, st_idx+1), ...]
						st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
						llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

						# t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
						t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
						h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
						w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
						llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
						st = ed + llm_grid_t * llm_grid_h * llm_grid_w
						# 现在的st是这段视频之后的text的起始位置

					else:
						token_positions = video_token_positions_split_by_frame[video_index] # (num_tokens, 7)
						# Each token_position is (Y_top, X_left, H, W, T_begin, T_end, window_index). Unit of Y and X are original pixels (should /= 32 to become token dimension); Unit of T is seconds.
						num_tokens, _ = token_positions.shape
						video_index += 1
						remain_videos -= 1
						ed = ed_video # 前一段text的结尾

						# 视频前面一段text的长度
						text_len = ed - st
						# rope_index of text section: [(st_idx, st_idx, st_idx), (st_idx+1, st_idx+1, st_idx+1), ...]
						st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
						llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

						# t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
						# TODO: 实际上应该在t_index加入真正的多元高精度timestamp（毕竟我们有）。但是考虑到模型训练的时候只用了t_index==0，怀疑用真的timestamp会损害性能。
						# TODO: h_index和w_index也应该尝试用float。
						t_index = torch.zeros(num_tokens)
						h_index = token_positions[:, 0] // 32
						w_index = token_positions[:, 1] // 32
						llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
						st = ed + num_tokens
						# 现在的st是这段视频之后的text的起始位置

				# Last text segment after the final image/video
				if st < len(input_tokens):
					st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
					text_len = len(input_tokens) - st
					llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

				llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
				position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(dtype=position_ids.dtype, device=position_ids.device)
				mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
			mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
			return position_ids, mrope_position_deltas
		else: # Pure text
			if attention_mask is not None:
				position_ids = attention_mask.long().cumsum(-1) - 1
				position_ids.masked_fill_(attention_mask == 0, 1)
				position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
				max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
				mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
			else:
				position_ids = (
					torch.arange(input_ids.shape[1], device=input_ids.device)
					.view(1, 1, -1)
					.expand(3, input_ids.shape[0], -1)
				)
				mrope_position_deltas = torch.zeros(
					[input_ids.shape[0], 1],
					device=input_ids.device,
					dtype=input_ids.dtype,
				)

			return position_ids, mrope_position_deltas

class EventVLForConditionalGeneration(Qwen3VLForConditionalGeneration):
	# This replaces __init__ of Qwen2_5_VLForConditionalGeneration
	def __init__(self, config, do_post_init=True):
		super(Qwen3VLForConditionalGeneration, self).__init__(config)
		self.model = EventVLModel(config, do_post_init=do_post_init)
		self.lm_head = torch.nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False, dtype=config.text_config.torch_dtype)
		
		if do_post_init:
			# Initialize weights and apply final processing
			self.post_init()

	@classmethod
	def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
		# 关键：加载时使用原始配置
		start_time = time.time()
		config = Qwen3VLConfig.from_pretrained(pretrained_model_name_or_path, dtype=kwargs.get('torch_dtype', None))

		from transformers.modeling_utils import no_init_weights
		# 使用 no_init_weights 跳过随机初始化，并使用 torch.device("cuda") 直接在 GPU 上创建模型
		with no_init_weights(), torch.device("cuda"):
			model = cls(config, do_post_init=False)

		print("Initialization wasted time: ", time.time() - start_time)

		qwen_state_dict = Qwen3VLForConditionalGeneration.from_pretrained(
			pretrained_model_name_or_path, 
			*args, 
			**kwargs
		).state_dict()

		# Check if lm_head is in the state_dict
		if "lm_head.weight" not in qwen_state_dict:
			raise ValueError(
				"The state_dict does not contain 'lm_head.weight'. "
				"Please ensure that the model is a Qwen2_5_VLForConditionalGeneration model."
			)

		# 从预训练文件加载权重
		model.load_state_dict(
			qwen_state_dict,
			strict=True
		)
		return model
	
	# Everything else is the same as Qwen3VLForConditionalGeneration, just provide positions instead of video_grid_thw
	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Cache] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		pixel_values: Optional[torch.Tensor] = None,
		pixel_values_videos: Optional[torch.FloatTensor] = None,
		image_grid_thw: Optional[torch.LongTensor] = None,
		video_token_positions: Optional[torch.Tensor] = None,
		tokens_per_video: Optional[torch.Tensor] = None,
		cache_position: Optional[torch.LongTensor] = None,
		logits_to_keep: Union[int, torch.Tensor] = 0,
		**kwargs: Unpack[TransformersKwargs],
	) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
			config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
			(masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
		image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
			The temporal, height and width of feature shape of each image in LLM.
		video_token_positions (`torch.FloatTensor` of shape `(num_video_tokens, 7)`, *optional*):
			For each video token, the corresponding [Y_top, X_left, H, W, T_begin, T_end, frame_index] is given. The x/y/h/w indexes are [TODO: What number?]

		Example:
			TODO: Add example
		"""
		outputs = self.model(
			input_ids=input_ids,
			pixel_values=pixel_values,
			pixel_values_videos=pixel_values_videos,
			image_grid_thw=image_grid_thw,
			video_token_positions=video_token_positions.to("cpu"),
			tokens_per_video=tokens_per_video.to("cpu"),
			position_ids=position_ids,
			attention_mask=attention_mask,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			cache_position=cache_position,
			**kwargs,
		)

		hidden_states = outputs[0]

		# Only compute necessary logits, and do not upcast them to float if we are not computing the loss
		slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
		logits = self.lm_head(hidden_states[:, slice_indices, :])

		loss = None
		if labels is not None:
			loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

		return Qwen3VLCausalLMOutputWithPast(
			loss=loss,
			logits=logits,
			past_key_values=outputs.past_key_values,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			rope_deltas=outputs.rope_deltas,
		)