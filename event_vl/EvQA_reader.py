import json
import glob
import os
import tqdm
from qwen_vl_utils import process_vision_info
import traceback
import torch
import random
import numpy as np
from model.e3vl_processor import process_vision_info_patched
from configs import EVQA_ROOT
from art_frt_free_inference import get_recons_video_paths

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EvQAReader:
	def __init__(self, data_path=EVQA_ROOT, lang="en", mode="text_only", video_root=None, seed=42, system_prompt=None):
		self.seed = seed # Reset for every question, to avoid randomness related to question sequence

		assert lang in ["en", "cn"] # English or Chinese
		assert mode in ["text_only", "video", "event"]

		if mode == "video" or mode == "event":
			assert video_root is not None
			self.video_root = video_root # e.g. annotate_system/video_cache/e2vid

		self.lang = lang
		self.mode = mode

		self.questions = []

		q_files = sorted(glob.glob(os.path.join(data_path, "questions", "*.json")))
		for q_file in q_files:
			with open(q_file, 'r') as f:
				data = json.load(f)
				#print(f"Loaded {len(data['questions'])} questions from {q_file}")
				self.questions.extend(data["questions"])

		if mode == "text_only":
			if lang == "en":
				self.system = "The following question is about a lost video. Based on knowledge and reasoning, guess the most likely answer and select the best option from the provided choices.\n"
			elif lang == "cn":
				self.system = "关于以下问题的视频资源已丢失。请基于知识和推理，猜测最有可能的答案，并从提供的选项中选择最佳选项。\n"
		elif mode == "video":
			if lang == "en":
				self.system = "You are a helpful assistant for answering questions about videos. This is a low quality black and white video reconstructed from event streams. Based on the video content and your knowledge, select the best option from the provided choices.\n"
			elif lang == "cn":
				self.system = "你是一个高水平的助手，专门回答有关视频的问题。这是一个从事件流重建的低质量黑白视频。请基于视频内容和你的知识，从提供的选项中选择最佳选项。\n"
		elif mode == "event":
			if lang == "en":
				self.system = "You are a helpful assistant for answering questions about videos. We have reconstructed a low quality black and white video from event streams, here are its key patches (not complete) in chronological order. Based on the video content and your knowledge, select the best option from the provided choices.\n"
			elif lang == "cn":
				self.system = "你是一个高水平的助手，专门回答有关视频的问题。我们从事件流中重建出了一个低质量黑白视频，这里是它的关键碎片（不全），按时间顺序排列。请基于视频内容和你的知识，从提供的选项中选择最佳选项。\n"
		if system_prompt is not None:
			# Override default system prompt
			self.system = system_prompt
		
		if lang == "en":
			self.question_prompt="\nOnly give the best option."
		elif lang == "cn":
			self.question_prompt="\n只需给出最佳选项。"

		if mode == "video" or mode == "event":
			# Make sure reconstructed videos are ready
			rec_mode = "frt" if mode == "video" else "art"
			h5_paths = [os.path.join(data_path, "h5_files", q["dataset_name"], q["file_path"]) for q in self.questions]
			seq_names = [q["dataset_name"] + "_" + str(q["question_id"]) for q in self.questions]
			get_recons_video_paths(mode=rec_mode, h5_paths=h5_paths, seq_names=seq_names, cache_dir=self.video_root)

	def __len__(self):
		return len(self.questions)

	def __getitem__(self, idx):
		question = self.questions[idx]

		system_prompt = self.system
		if self.lang == "en":
			question_text = f"Question: {question['question']}\nOptions:\n(A) {question['choices'][0]}\n(B) {question['choices'][1]}\n(C) {question['choices'][2]}\n(D) {question['choices'][3]}\n"
			# correct_answer should be A/B/C/D
			correct_answer = chr(ord('A') + question['choices'].index(question["answer"]))

		elif self.lang == "cn":
			question_text = f"问题: {question['question_cn']}\n选项:\n(A) {question['choices_cn'][0]}\n(B) {question['choices_cn'][1]}\n(C) {question['choices_cn'][2]}\n(D) {question['choices_cn'][3]}\n"
			correct_answer = chr(ord('A') + question['choices_cn'].index(question["answer_cn"]))

		user_prompt = question_text + self.question_prompt

		return system_prompt, user_prompt, question, correct_answer
	
	def answer_question(self, question_idx, processor, model, fps, output_file="results.txt", tokens_per_frame=500, only_return_input_token_cnt=False):
		system_prompt, user_prompt, data_sample, correct_answer = self[question_idx]
		set_seed(self.seed)

		user_content = []
		if self.mode == "video":
			video_path = f"{self.video_root}/{data_sample['dataset_name']}_{data_sample['question_id']}.mp4"
			assert os.path.exists(video_path), f"Video not found: {video_path}"	
			user_content.append({
				"type": "video",
				"video": video_path,
				"fps": fps,
			})
		elif self.mode == "event":
			video_path = f"{self.video_root}/{data_sample['dataset_name']}_{data_sample['question_id']}.npz"
			assert os.path.exists(video_path), f"Event video not found: {video_path}"
			user_content.append({
				"type": "video",
				"video": video_path,
			})

		user_content.append({
			"type": "text",
			"text": user_prompt
		})

		messages = [
			{
				"role": "system",
				"content": [
					{
						"type": "text",
						"text": system_prompt
					}
				]
			},
			{
				"role": "user",
				"content": user_content
			}
		]

		text = processor.apply_chat_template(
			messages, tokenize=False, add_generation_prompt=True
		)
		'''
		The qwen processor makes the text like this:
		<|im_start|>system
		You are a helpful assistant.<|im_end|>
		<|im_start|>user
		<|vision_start|><|video_pad|><|vision_end|>Describe the video.<|im_end|>
		<|im_start|>assistant
		'''
		answer_prompt = "Best option:("
		# Add the answer prompt to guide formatting, like with VideoChat.
		text = text + answer_prompt
		#print(text)

		if self.mode in ["text_only", "video"]:
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
		elif self.mode == "event":
			images, videos_patched = process_vision_info_patched(messages, image_patch_size=16)

			inputs = processor(
				text=text,
				images=images,
				patch_videos=videos_patched,
				return_tensors="pt",
				do_resize=False,
				tokens_per_frame=tokens_per_frame,#(596//32)*(336//32),
			)

		if "pixel_values_videos" in inputs:
			visual_token_cnt = inputs["pixel_values_videos"].shape[0] // 4
		else:
			visual_token_cnt = 0

		input_token_cnt = inputs.input_ids.shape[1]

		if only_return_input_token_cnt:
			return input_token_cnt
		
		inputs = inputs.to("cuda")

		# --- Start of Measurement ---
		torch.cuda.reset_peak_memory_stats()

		generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

		peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
		# --- End of Measurement ---

		generated_ids_trimmed = [
			out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
		]
		generated_token_cnt = len(generated_ids_trimmed[0])

		output_text = processor.batch_decode(
			generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
		)
		#print("output_text:", output_text)

		# remove potential explanation
		return_prompt='('
		output_text = return_prompt + output_text[0].strip().split('\n')[0]
		output_answer = output_text[1].upper() # B from (B) xxx
		is_correct = (output_answer == correct_answer)

		with open(output_file, 'a', encoding="UTF-8") as f:
			f.write(f"{data_sample['qid']} {is_correct} {input_token_cnt} {output_text}\n")
		
		return output_text, is_correct, visual_token_cnt, input_token_cnt, generated_token_cnt, peak_memory_mb
	
	def sort_output_file(self, output_file="results.txt"):
		# Read all lines from the output file. Sort them so the qids are in the order of self.questions.
		if not os.path.exists(output_file):
			print(f"Output file {output_file} does not exist. Cannot sort.")
			return
		with open(output_file, 'r', encoding="UTF-8") as f:
			lines = f.readlines()
		qid_to_line = {}
		for line in lines:
			parts = line.strip().split(' ')
			if len(parts) >= 1:
				qid = parts[0]
				qid_to_line[qid] = line
		sorted_lines = []
		for q in self.questions:
			qid = q['qid']
			if qid in qid_to_line:
				sorted_lines.append(qid_to_line[qid])
			else:
				print(f"Warning: qid {qid} not found in output file.")
		with open(output_file, 'w', encoding="UTF-8") as f:
			f.writelines(sorted_lines)

	def evaluate_all(self, processor, model, fps=None, output_file="results.txt", skip_answered=False, tokens_per_frame=500):
		correct_count = 0
		total_visual_tokens = 0
		total_input_tokens = 0
		total_generated_tokens = 0
		total_peak_memory = 0
		max_peak_memory = 0
		max_total_tokens = 0
		failure_count = 0

		previous_correct_count = 0
		# If skip_answered is True, read from output_file the already answered qids. Skip those in evaluation.
		answered_qids = set()
		if skip_answered and os.path.exists(output_file):
			previous_correct = {}
			new_lines = []
			with open(output_file, 'r', encoding="UTF-8") as f:
				lines = f.readlines()

			# Create a map from qid to question data for quick lookup
			qid_to_question = {q['qid']: q for q in self.questions}

			for line in lines:
				parts = line.strip().split(' ')
				if len(parts) >= 2:
					qid = parts[0]
					is_correct_str = parts[1]
					
					# Check if the line is in the old format (qid is_correct output_text)
					# by checking if the third part is not a number.
					is_old_format = True
					if len(parts) > 2:
						try:
							int(parts[2])
							is_old_format = False
						except ValueError:
							is_old_format = True
					
					if is_old_format:
						print(f"Found old format result for qid {qid}. Calculating token count.")
						input_token_cnt = self.answer_question(
							question_idx=self.questions.index(qid_to_question[qid]),
							processor=processor,
							model=model,
							fps=fps,
							output_file=output_file,
							only_return_input_token_cnt=True
						)
						
						output_text = ' '.join(parts[2:])
						new_line = f"{qid} {is_correct_str} {input_token_cnt} {output_text}\n"
						new_lines.append(new_line)
					else:
						# New format, keep as is
						new_lines.append(line)

					answered_qids.add(qid)
					correct = is_correct_str == 'True'
					previous_correct[qid] = correct
			
			# Write the updated lines back to the file
			with open(output_file, 'w', encoding="UTF-8") as f:
				f.writelines(new_lines)

			print(f"Skipping {len(answered_qids)} already answered questions.")
			previous_correct_count = sum(1 for v in previous_correct.values() if v)
			
		pbar = tqdm.tqdm(range(len(self)))
		for i in pbar:
			try:
				if skip_answered and self.questions[i]['qid'] in answered_qids:
					continue
				_, is_correct, visual_token_cnt, input_token_cnt, generated_token_cnt, peak_memory_mb = self.answer_question(i, processor, model, fps=fps, output_file=output_file, tokens_per_frame=tokens_per_frame)
				if is_correct:
					correct_count += 1
				
				# Accumulate stats
				total_visual_tokens += visual_token_cnt
				total_input_tokens += input_token_cnt
				total_generated_tokens += generated_token_cnt
				total_peak_memory += peak_memory_mb
				
				# Update maximums
				if peak_memory_mb > max_peak_memory:
					max_peak_memory = peak_memory_mb
				current_total_tokens = input_token_cnt + generated_token_cnt
				if current_total_tokens > max_total_tokens:
					max_total_tokens = current_total_tokens
				
				# Update progress bar
				processed_count = i + 1
				wrong_count = processed_count - correct_count
				accuracy = correct_count / processed_count
				avg_input_tokens = total_input_tokens / processed_count
				avg_generated_tokens = total_generated_tokens / processed_count
				avg_peak_memory = total_peak_memory / processed_count
				pbar.set_postfix(
					accuracy=f"{accuracy:.2%}", 
					avg_mem_MB=f"{avg_peak_memory:.2f}",
					max_mem_MB=f"{max_peak_memory:.2f}",
					max_tokens=max_total_tokens
				)
			except Exception as e:
				# Print the error stack trace for debugging
				traceback.print_exc()
				print(f"Error processing question {self.questions[i]['qid']}: {e}")
				failure_count += 1
				continue
		
		total_questions = len(self) - failure_count
		if total_questions == 0:
			print("No questions were processed successfully.")
			return 0, 0, 0, 0, 0, failure_count, "No successful runs."

		correct_count += previous_correct_count
		accuracy = correct_count / total_questions
		avg_input_tokens = total_input_tokens / total_questions
		avg_generated_tokens = total_generated_tokens / total_questions
		avg_peak_memory = total_peak_memory / total_questions
		
		message = (
			f"Accuracy: {accuracy*100:.2f}% ({correct_count}/{total_questions}), "
			f"Avg Input Tokens: {avg_input_tokens:.2f}, "
			f"Avg Generated Tokens: {avg_generated_tokens:.2f}, "
			f"Avg Peak Memory: {avg_peak_memory:.2f} MB, "
			f"Max Peak Memory: {max_peak_memory:.2f} MB, "
			f"Max Total Tokens: {max_total_tokens}, "
			f"Failures: {failure_count}",
			f" (including {previous_correct_count} correct from previous runs)" if previous_correct_count > 0 else ""
		)
		print(message)

		if skip_answered:
			self.sort_output_file(output_file=output_file)

		return accuracy, avg_input_tokens, avg_generated_tokens, avg_peak_memory, max_peak_memory, max_total_tokens, failure_count, message