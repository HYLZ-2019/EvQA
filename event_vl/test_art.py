import os
import sys
import torch
import time
from datetime import datetime

from model.e3vl_model import EventVLForConditionalGeneration
from model.e3vl_processor import EventVLProcessor
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

def do_experiment(npz_root, qwen_name):
	begin_time = time.time()

	# qwen_name: "4", find path from qwen_paths list
	qwen_path = None
	for name, size in qwen_paths:
		if size == qwen_name:
			qwen_path = os.path.join(QWEN_ROOT, name)
			break
	if qwen_path is None:
		print(f"Qwen model with size {qwen_name}B not found.")
		return False
	experiment_name = os.path.basename(npz_root)

	qa_reader = EvQAReader(data_path=EVQA_ROOT, lang="en", mode="event", video_root=npz_root)
	processor = EventVLProcessor.from_pretrained(qwen_path)

	model = None
	load_model = True
	if load_model:
		model = EventVLForConditionalGeneration.from_pretrained(
			qwen_path,
			dtype=torch.bfloat16,
			attn_implementation="flash_attention_2",
			device_map="auto",
		)

	# Update the unpacking of return values from evaluate_all
	accuracy, avg_input_tokens, avg_generated_tokens, avg_peak_memory, max_peak_memory, max_total_tokens, failure_count, message = qa_reader.evaluate_all(
		processor, model, output_file=f"results/art_{experiment_name}_Q{qwen_name}B_Thinking.txt", skip_answered=True
	)

	end_time = time.time()

	success_str = "succeeded" if failure_count == 0 else f"had {failure_count} failures"
	# Begin time, end time, duration; format YYYY-MM-DD HH:MM:SS
	output_str = f"Experiment for ART {experiment_name} with Qwen {qwen_name}B {success_str}: {datetime.fromtimestamp(begin_time).strftime('%Y-%m-%d %H:%M:%S')} to {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}, Duration: {end_time - begin_time:.2f} seconds. Results: {message}\n"
	print(output_str)

	# Append into the log file for summary
	with open("results/summary_log.txt", "a", encoding="UTF-8") as f:
		f.write(output_str)
	
	del model
	torch.cuda.empty_cache()


if __name__ == "__main__":
	# parse command line arguments, first is npz_root, second is qwen_name
	if len(sys.argv) >= 3:
		npz_root = sys.argv[1]
		if npz_root[-1] == "/":
			npz_root = npz_root[:-1]
		print("Beginning testing for", os.path.basename(npz_root))
		qwen_name = sys.argv[2]
		do_experiment(npz_root, qwen_name)
	else:
		print("Usage: python test_art.py <npz_root> <qwen_name>")
		print("Example: python test_art.py /path/to/art_dataset 4")

		qwen_name = "2"
		npz_root = "video_cache/art_inference"

		# Make reconstructed npz videos for all sequences
		do_experiment(npz_root, qwen_name)