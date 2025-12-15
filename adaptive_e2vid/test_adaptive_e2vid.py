import os
import sys
import yaml
import torch
import tqdm
import cv2
import numpy as np
from collections import defaultdict
from utils.data import data_sources

from data.data_interface import make_concat_multi_dataset
from torch.utils.data import DataLoader
from model.sparse_e2vid_interface import SparseE2VIDInterface
from collections import OrderedDict
from utils.metric_references import beat_method
from train import convert_to_compiled
from model.sparse_e2vid_interface import unpack_dataset
from model.train_utils import load_raft, inference_raft, norm, printshapes, nan_hook, normalize, concat_imgs, normalize_nobias, normalize_batch_voxel
from utils.util import instantiate_from_config, instantiate_scheduler_from_config, get_obj_from_str

# 强健处理 pred 维度（确保为 torch.Tensor 在 CPU 上）
def to_cpu_uint8(t):
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
    else:
        t = torch.tensor(t).cpu()
    # remove batch/channel singletons if any
    t = t.squeeze()
    if t.ndim == 3:  # (C,H,W) -> take first channel
        t = t[0]
    if t.ndim != 2:
        raise ValueError(f"Unexpected pred shape after squeeze: {t.shape}")
    return t.numpy().astype(np.uint8)

def create_test_dataloader(stage_cfg):
	dataset = make_concat_multi_dataset(stage_cfg["test"])
	dataset_collate_fn = stage_cfg["dataset_collate_fn"]
	if dataset_collate_fn is not None:
		collate_fn = get_obj_from_str(dataset_collate_fn)
	else:
		collate_fn = None
	dataloader = DataLoader(dataset,
							batch_size=1,
							num_workers=stage_cfg["test_num_workers"],
							shuffle=False,
							collate_fn=collate_fn)
	return dataloader

def run_test(model_interface, dataloader, device, configs):

    output_dir = configs["test_output_dir"]
    save_in_pairs = configs.get("save_in_pairs", True)
    
    model_interface.e2vid_model.eval()
    
    previous_test_sequence = None

    # collect failures
    failed_samples = []

    with torch.no_grad():

        for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
            # try/except per-sample to skip failures (OOM / other errors)
            try:
                sequence_name = batch["sequence_name"]
                #print(f"Testing sequence: {sequence_name}, batch {batch_idx+1}/{len(dataloader)}")
                # model_interface.e2vid_model.reset_states()
                if output_dir is not None:
                    dataset_name = batch["dataset_name"]
                    datatset_output_dir = os.path.join(output_dir, dataset_name)
                    output_npz_dir = os.path.join(datatset_output_dir, "npz_files")
                    output_mp4_dir = os.path.join(datatset_output_dir, "mp4_videos")
                    os.makedirs(output_npz_dir, exist_ok=True)
                    os.makedirs(output_mp4_dir, exist_ok=True)
                npz_filename = os.path.join(output_npz_dir, f"{sequence_name}.npz")
                if os.path.exists(npz_filename):
                    #print(f"Output npz file {npz_filename} already exists, skipping...")
                    continue

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(device)
                
                data = batch # 由于特殊的no_collate_function，batch本身就是一个样本
                #print("batch_shape",{k:v.shape for k,v in batch.items() if torch.is_tensor(v)})
                pred = model_interface.forward_sequence(batch,reset_states=True,val=False)
                patches = unpack_dataset(data)

                H = data["H"]
                W = data["W"]
                T = len(patches)

                # 准备.npz格式的输出数据
                shapes = []
                flat_data = []
                indices = []
                positions = []
                current_index = 0

                if save_in_pairs:
                    # 创建相邻的patch对（每两个连续的patch组成一对）
                    for i in range(0, T-1, 2):  # 步长为2，确保每对都是相邻的patch
                        if i + 1 >= T:  # 确保有足够的patch组成对
                            break
                            
                        # 获取当前patch对
                        patch1 = patches[i]
                        patch2 = patches[i+1]
                        
                        # 获取对应的预测结果
                        pred1 = pred[i]["pred"]*255  # 第一个时间帧的预测
                        pred2 = pred[i+1]["pred"]*255  # 第二个时间帧的预测

                        p1 = to_cpu_uint8(pred1)
                        p2 = to_cpu_uint8(pred2)

                        # 将两个时间帧堆叠在一起 (H, W, 2)
                        patch_pair_np = np.stack([p1, p2], axis=-1)  # 形状: (H, W, 2)

                        # 记录shape
                        shapes.append([p1.shape[0], p1.shape[1], 2])
                        
                        # 扁平化数据并记录索引
                        flat_patch = patch_pair_np.flatten()
                        flat_data.append(flat_patch)
                        
                        begin_idx = current_index
                        end_idx = current_index + len(flat_patch)
                        indices.append([begin_idx, end_idx])
                        current_index = end_idx
                        
                        # 记录位置信息 (y, x, h, w, begin_t, end_t)
                        y = patch1["ey"]
                        x = patch1["ex"]
                        h = patch1["eh"]
                        w = patch1["ew"]
                        begin_t = patch1["begin_t"]
                        end_t = patch2["end_t"]  # 使用第二个patch的结束时间
                        
                        positions.append([y, x, h, w, begin_t, end_t])
                    
                else:
                    # 基本相同的处理方式，但是不合并pairs，保存的shape也都是二维的
                    for i in range(T):
                        patch = patches[i]
                        pred_img = pred[i]["pred"]*255
                        p = to_cpu_uint8(pred_img)

                        # 记录shape
                        shapes.append([p.shape[0], p.shape[1]])
                        
                        # 扁平化数据并记录索引
                        flat_patch = p.flatten()
                        flat_data.append(flat_patch)
                        
                        begin_idx = current_index
                        end_idx = current_index + len(flat_patch)
                        indices.append([begin_idx, end_idx])
                        current_index = end_idx
                        
                        # 记录位置信息 (y, x, h, w, begin_t, end_t)
                        y = patch["ey"]
                        x = patch["ex"]
                        h = patch["eh"]
                        w = patch["ew"]
                        begin_t = patch["begin_t"]
                        end_t = patch["end_t"]
                        
                        positions.append([y, x, h, w, begin_t, end_t])

                # 将所有扁平化的数据连接起来并保存为 npz
                if flat_data:
                    flat_data_concatenated = np.concatenate(flat_data)
                    shapes_np = np.array(shapes)
                    indices_np = np.array(indices)
                    positions_np = np.array(positions)
                    
                    np.savez(npz_filename, 
                                shapes=shapes_np,
                                flat_data=flat_data_concatenated,
                                indices=indices_np,
                                positions=positions_np)
                else:
                    print("No patch pairs to save")

                # 以下为视频写入部分（保持原样）
                output_filename = os.path.join(output_mp4_dir, f"{sequence_name}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_rate = 30
                frame_duration = 1000000 // frame_rate  # 每帧的微秒数

                # 找到最小和最大时间
                et = [patch['end_t']*1e6 for patch in patches]

                # 创建视频写入器
                vid = cv2.VideoWriter(output_filename, fourcc, frame_rate, (W*2, H))

                if not vid.isOpened():
                    print("Error: Could not open video writer")
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    vid = cv2.VideoWriter(output_filename, fourcc, frame_rate, (W*2, H))
                    if not vid.isOpened():
                        print("Error: Could not open video writer with MJPG codec either")
                        exit(1)

                # 初始化当前帧和patch索引
                current_time = et[0]
                patch_index = 0
                total_frames = int((et[-1] - et[0]) // frame_duration + 1)

                # 创建初始图像
                img = np.zeros((H, W*2), dtype=np.uint8)

                # 按照时间顺序处理所有帧
                for frame_count in range(total_frames):
                    # 处理所有在当前时间范围内的patch
                    while patch_index < T and et[patch_index] <= current_time:
                        patch_data = patches[patch_index]
                        
                        events = patch_data["voxel"]
                        pred_img = pred[patch_index]["pred"]*255
                        ey = patch_data["ey"]
                        ex = patch_data["ex"]
                        eh = patch_data["eh"]
                        ew = patch_data["ew"]
                        
                        # 事件可视化
                        event_vis = normalize_nobias(torch.sum(events, dim=0, keepdim=True)) * 255  # 1, eh, ew
                        
                        # 将事件可视化转换为numpy数组
                        event_vis_np = event_vis.squeeze().cpu().numpy().astype(np.uint8)
                        
                        # 将patch放到正确的位置
                        img[ey:ey+eh, ex:ex+ew] = event_vis_np
                        img[ey:ey+eh, W+ex:W+ex+ew] = pred_img.cpu().numpy().astype(np.uint8)
                        
                        patch_index += 1

                    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    vid.write(img_color)
                    
                    # 更新时间到下一帧
                    current_time += frame_duration
                    
                vid.release()
                print(f"Video saved to {output_filename}")

            # end try
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # record failure and skip this sample
                try:
                    seq_str = str(batch.get("sequence_name", "unknown"))
                except Exception:
                    seq_str = f"idx_{batch_idx}"
                print(f"Error processing sample {seq_str}: {e}. Skipping.")
                # free cuda memory if any and do a full gc
                try:
                    import gc
                    gc.collect()
                except Exception:
                    pass
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                failed_samples.append((seq_str, repr(e)))
                continue
    
    # end with torch.no_grad()

    # Optionally write failed samples to a file for later inspection
    fail_log = os.path.join("tensorboard_logs", configs["experiment_name"], "failed_samples.txt")
    os.makedirs(os.path.dirname(fail_log), exist_ok=True)
    with open(fail_log, "w", encoding="utf-8") as f:
        if failed_samples:
            for s, err in failed_samples:
                f.write(f"{s}\t{err}\n")
            print(f"Saved failed sample list to {fail_log}")
        else:
            f.write("No failures\n")
            print("No failures during test.")
def main():
	# Add two arguments.
	# Argument 1: config_path
	# Argument 2 (optional): test_all_pths (default=False)
	if len(sys.argv) > 1:
		config_path = sys.argv[1]
	else:
		config_path = "configs/template.yaml"

	if len(sys.argv) > 2:
		test_all_pths = True
	else:
		test_all_pths = False

	with open(config_path) as f:
		config = yaml.load(f, Loader=yaml.Loader)

	assert config.get("task", "sparse_e2vid") == "sparse_e2vid", "flow should be tested with test_flow.py"

	ckpt_paths_file = f"ckpt_paths/{config['experiment_name']}.txt"
	output_csv = os.path.join("tensorboard_logs", config['experiment_name'], f"all_test_results_new.csv")
	os.makedirs(os.path.dirname(output_csv), exist_ok=True)
	done_checkpoints = []
	if os.path.exists(output_csv):
		with open(output_csv, "r", encoding="utf-8") as f:
			lines = f.readlines()
			for line in lines[1:]:
				ckpt_path = line.split(",")[0]
				done_checkpoints.append(ckpt_path)

	# all_results = []
	if os.path.exists(ckpt_paths_file) and os.path.getsize(ckpt_paths_file) > 0:
		with open(ckpt_paths_file, "r") as f:
			paths = [p.strip() for p in f.readlines() if p.strip()]
			assert len(paths) > 0, "No checkpoint paths found in the file."
			if not test_all_pths:
				paths = paths[-1:]

			for path in paths:
				subpath = path.split("/")[-1]
				# If I only request testing the last line, don't skip, it is probably retesting
				if not test_all_pths or subpath not in done_checkpoints:
					run_single_test(path, config)
					# all_results.append((result, subpath))

	else:
		print("No checkpoint paths file found or it is empty.")

def run_single_test(checkpoint_path, config):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model_interface = SparseE2VIDInterface(config["module"], device=device, local_rank=None)

	
	if checkpoint_path is not None:
		saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
		state_dict = saved["state_dict"]
		
		# Don't use torch.compile, because the test is fast enough.
		new_state_dict = convert_to_compiled(state_dict=state_dict, local_rank=None, use_compile=False)
		
		model_interface.e2vid_model.load_state_dict(new_state_dict, strict=False)
		print("Loaded checkpoint:", checkpoint_path)

	model_interface.e2vid_model.to(device)

	test_dataloader = create_test_dataloader(config["test_stage"])
	run_test(model_interface, test_dataloader, device, config)

if __name__ == "__main__":
	main()
