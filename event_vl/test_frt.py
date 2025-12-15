import os
import sys
import time
from transformers import AutoModelForImageTextToText, AutoProcessor
from datetime import datetime
import torch
from EvQA_reader import EvQAReader
from configs import QWEN_ROOT, EVQA_ROOT

qwen_paths = [
	("Qwen3-VL-2B-Thinking", "2"),
	("Qwen3-VL-4B-Thinking", "4"),
	("Qwen3-VL-8B-Thinking", "8"),
	("Qwen3-VL-30B-A3B-Thinking", "30"),
	("Qwen3-VL-32B-Thinking", "32"),
	("Qwen3-VL-235B-A22B-Thinking", "235"),
]

def do_experiment(video_root, fps, qwen_name):
	# qwen_name: "4", find path from qwen_paths list
	qwen_path = None
	for name, size in qwen_paths:
		if size == qwen_name:
			qwen_path = os.path.join(QWEN_ROOT, name)
			break
	if qwen_path is None:
		print(f"Qwen model with size {qwen_name}B not found.")
		return False
	
	begin_time = time.time()

	qa_reader = EvQAReader(data_path=EVQA_ROOT, lang="en", mode="video", video_root=video_root)
	procesor = AutoProcessor.from_pretrained(qwen_path)
	model = AutoModelForImageTextToText.from_pretrained(qwen_path, dtype="auto", device_map="auto")
	# Update the unpacking of return values from evaluate_all
	accuracy, avg_input_tokens, avg_generated_tokens, avg_peak_memory, max_peak_memory, max_total_tokens, failure_count, message = qa_reader.evaluate_all(
		procesor, model, fps=fps, output_file=f"results/video_{fps}fps_Q{qwen_name}B_Thinking.txt", skip_answered=True
	)

	end_time = time.time()

	success_str = "succeeded" if failure_count == 0 else f"had {failure_count} failures"
	# Begin time, end time, duration; format YYYY-MM-DD HH:MM:SS
	output_str = f"Experiment for Qwen {qwen_name}B at {fps} FPS {success_str}: {datetime.fromtimestamp(begin_time).strftime('%Y-%m-%d %H:%M:%S')} to {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}, Duration: {end_time - begin_time:.2f} seconds. Results: {message}\n"
	print(output_str)

	# Append into the log file for summary
	with open("results/summary_log.txt", "a", encoding="UTF-8") as f:
		f.write(output_str)
	
	del model
	torch.cuda.empty_cache()
	
	return True

if __name__ == "__main__":
	# parse command line arguments, first is video_root, second is qwen_name, third is fps
	if len(sys.argv) >= 4:
		video_root = sys.argv[1]
		if video_root[-1] == "/":
			video_root = video_root[:-1]
		print("Beginning testing for", os.path.basename(video_root))
		qwen_name = sys.argv[2]
		fps = float(sys.argv[3])
		do_experiment(video_root, fps, qwen_name)
	else:
		print("Usage: python test_art.py <video_root> <qwen_name> <fps>")
		print("Example: python test_art.py /path/to/art_dataset 4 1")

		qwen_name = "2"
		FPS = 2
		video_root = "video_cache/frt_inference"
		# Make reconstructed npz videos for all sequences
		do_experiment(video_root, FPS, qwen_name)