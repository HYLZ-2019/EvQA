import glob
import os
import sys
from model.e3vl_processor import process_vision_info_patched
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor
from model.e3vl_model import EventVLForConditionalGeneration
from model.e3vl_processor import EventVLProcessor
import torch
from configs import QWEN_ROOT, EVQA_ROOT

# Add the parent directory to sys.path to enable imports from sibling directories
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from adaptive_e2vid import VideoReconstructor, VideoReconstructor_Adaptive

def get_path(path_list):
	for p in path_list:
		if os.path.exists(p):
			return p
	assert False, "None of the paths exist."

def inference_art(model, processor, question, npz_path):
	system_prompt = "You are a helpful assistant for answering questions about videos. We have reconstructed a low quality black and white video from event streams, here are its key patches (not all patches) in chronological order. Objects may appear multiple times in one frame, corresponding to their status at different times. Based on the shattered video content and your knowledge, answer the user's question. The user already knows that the video is black and white with low quality and shattered into pieces, so there's no need to mention this. Keep the answer concise.\n"
	messages = [
		{"role": "system", "content": [{"type": "text", "text": system_prompt}]},
		{"role": "user", "content": [
			{"type": "text", "text": question},
			{"type": "video", "video": npz_path},
		]}
	]
	text = processor.apply_chat_template(
		messages, tokenize=False, add_generation_prompt=True
	)
	images, videos_patched = process_vision_info_patched(messages, image_patch_size=16)
	tokens_per_frame = 512
	inputs = processor(
		text=text,
		images=images,
		patch_videos=videos_patched,
		return_tensors="pt",
		do_resize=False,
		tokens_per_frame=tokens_per_frame,
	)
	inputs = inputs.to("cuda")
	generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)

	generated_ids_trimmed = [
		out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
	]

	output_text = processor.batch_decode(
		generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
	)
	
	return output_text[0]

def inference_frt(model, processor, question, mp4_path):
	system_prompt = "You are a helpful assistant for answering questions about videos. This is a low quality black and white video reconstructed from event streams, where backgrounds may be black, white or blurry due to lack of information, so ignore the backgrounds when they have low quality. Based on the video content and your knowledge, answer the user's question. The user already knows that the video is black and white with low quality, so there's no need to mention this. Keep the answer concise.\n"
	messages = [
		{"role": "system", "content": [{"type": "text", "text": system_prompt}]},
		{"role": "user", "content": [
			{"type": "text", "text": question},
			{"type": "video", "video": mp4_path},
		]}
	]
	text = processor.apply_chat_template(
		messages, tokenize=False, add_generation_prompt=True
	)
	images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

	# split the videos and according metadatas
	if videos is not None:
		videos, video_metadatas = zip(*videos)
		videos, video_metadatas = list(videos), list(video_metadatas)
	else:
		video_metadatas = None

	inputs = processor(
		text=text,
		images=images,
		videos=videos,
		video_metadata=video_metadatas,
		return_tensors="pt",
		do_resize=False,
		**video_kwargs
	)
	inputs = inputs.to("cuda")
	generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)

	generated_ids_trimmed = [
		out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
	]

	output_text = processor.batch_decode(
		generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
	)
	
	return output_text[0]

qwen_paths = [
	("Qwen3-VL-2B-Thinking", "2"),
	("Qwen3-VL-4B-Thinking", "4"),
	("Qwen3-VL-8B-Thinking", "8"),
	("Qwen3-VL-30B-A3B-Thinking", "30"),
	("Qwen3-VL-32B-Thinking", "32"),
	("Qwen3-VL-235B-A22B-Thinking", "235"),
]

def inference_batch(mode, qwen_name, input_list, output_file="outputs/inference_results.txt"):
	qwen_path = None
	for name, size in qwen_paths:
		if size == qwen_name:
			qwen_path = os.path.join(QWEN_ROOT, name)
			break
	if qwen_path is None:
		print(f"Qwen model with size {qwen_name}B not found.")
		return False

	if mode == "frt":
		procesor = AutoProcessor.from_pretrained(qwen_path)
		model = AutoModelForImageTextToText.from_pretrained(qwen_path, dtype="auto", device_map="auto")
	elif mode == "art":
		processor = EventVLProcessor.from_pretrained(qwen_path)
		model = EventVLForConditionalGeneration.from_pretrained(
			qwen_path,
			dtype=torch.bfloat16,
			attn_implementation="flash_attention_2",
			device_map="auto",
		)
	
	for question, data_path in input_list:
		if mode == "frt":
			output = inference_frt(model, procesor, question, data_path)
		else:
			output = inference_art(model, processor, question, data_path)
		out_str = f"======\nQuestion: {question}\nData: {data_path}\nAnswer:\n{output}\n\n"
		with open(output_file, "a", encoding="UTF-8") as f:
			f.write(out_str)
		print(out_str)

def get_recons_video_paths(mode, h5_paths, seq_names, cache_dir):
	# Generate the reconstructed mp4/npz video from h5_path, and save to cache_dir/{seq_name}.{mp4/npz}.
	# If already exists, directly return the path.
	assert mode in ["frt", "art"]
	os.makedirs(cache_dir, exist_ok=True)

	attr = "mp4" if mode == "frt" else "npz"
	video_paths = [os.path.join(cache_dir, f"{name}.{attr}") for name in seq_names]
	
	# Check existence
	vids_to_recon = []
	for h5_path, seq_name, vid_path in zip(h5_paths, seq_names, video_paths):
		if not os.path.exists(vid_path):
			vids_to_recon.append((h5_path, seq_name, vid_path))
	if len(vids_to_recon) == 0:
		# All exist
		return video_paths
	
	if mode == "frt":
		reconstructor = VideoReconstructor()
		for h5_path, seq_name, video_path in vids_to_recon:
			print(f"Reconstructing FRT video for {seq_name}...")
			reconstructor.h5_to_video(h5_path, video_path, FPS=24)
	else:
		reconstructor = VideoReconstructor_Adaptive()
		for h5_path, seq_name, video_path in vids_to_recon:
			print(f"Reconstructing ART video for {seq_name}...")
			reconstructor.h5_to_video(h5_path, video_path)
	return video_paths


if __name__ == "__main__":
	
	test_list = [
		("PEDRo", "rec70_02764_02777", "Write a short poem inspired by the content of the video."),
	]
	h5_paths = [
		os.path.join(EVQA_ROOT, "h5_files", dataset_name, question_id + ".h5") for dataset_name, question_id, _ in test_list
	]
	sequence_names = [f"{dataset_name}_{question_id}" for dataset_name, question_id, _ in test_list]


	frt_cache_path = "video_cache/frt_inference"
	art_cache_path = "video_cache/art_inference"

	frt_video_paths = get_recons_video_paths("frt", h5_paths, sequence_names, frt_cache_path)
	art_video_paths = get_recons_video_paths("art", h5_paths, sequence_names, art_cache_path)

	frt_list = [
		(question_text, frt_video_path) for (_, _, question_text), frt_video_path in zip(test_list, frt_video_paths)
	]
	art_list = [
		(question_text, art_video_path) for (_, _, question_text), art_video_path in zip(test_list, art_video_paths)
	]

	os.makedirs("results", exist_ok=True)
	inference_batch("frt", "2", frt_list, output_file="results/frt_inference_results.txt")
	inference_batch("art", "2", art_list, output_file="results/art_inference_results.txt")