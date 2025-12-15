# Reconstruction as a Bridge for Event-Based Visual Question Answering

This is the codebase for the paper ["Reconstruction as a Bridge for Event-Based Visual Question Answering"](https://arxiv.org/pdf/2512.11510). The proposed EvQA dataset can be downloaded from Huggingface(https://huggingface.co/datasets/hylz/EvQA), while this repository includes:

- A system used to review questions from the EvQA dataset;
- Inference code for the proposed methods FRT & ART; and
- Training code for the Adaptive-E2VID model used by ART.

## Dataset format and download

The EvQA benchmark can be downloaded from Huggingface (https://huggingface.co/datasets/hylz/EvQA). The dataset is organized into two main directories: `h5_files` and `questions`. Let `{evqa_root}` be the root directory of the EvQA dataset so that `{evqa_root}/questions/` exists.

The `questions` directory contains the annotation files in JSON format, named as `questions/{dataset_name}.json`. Each JSON file contains metadata for the dataset and a list of questions. The metadata includes the dataset name and description. Each entry in the question list contains the question ID, file path to the corresponding H5 file, camera type, resolution, question type, duration, keywords, and text for the questions, answers and multiple-choice options (in both English and Chinese).

The `h5_files` directory contains the event data files and license information, organized by dataset name. Specifically, for each dataset, the structure is `h5_files/{dataset_name}/{question_id}.h5` and `h5_files/{dataset_name}/LICENSE.txt`. 

The event data is stored in HDF5 (`.h5`) format, generated using `h5py` with compression enabled. Each H5 file contains the event stream and optional image frames. 

The file attributes store essential metadata, including:
- `sensor_resolution`: The spatial resolution (H, W) of the sensor.
- `num_events`: The total number of events.
- `num_imgs`: The number of intensity images (0 if unavailable).
- `duration`: The duration of the recording in seconds.
- `camera_type`: The model of the event camera.
- `data_source`: The name of the source dataset.
- `base_time`: The absolute timestamp (epoch time) of the start of the recording. If the original timestamps are relative, this is set to 0.0. The timestamps in the file are relative to this base time (i.e., starting from 0).

The events are stored separately within the H5 file:
- `events/xs`: X-coordinates of the events (`uint16`).
- `events/ys`: Y-coordinates of the events (`uint16`).
- `events/ts`: Timestamps of the events in microseconds (`uint64`), with `ts[0] == 0`. We use `uint64` to avoid overflow for recordings longer than approximately 1.19 hours.
- `events/ps`: Polarity of the events (`uint8`), taking values of 0 or 1.

If intensity images are available, they are stored as separate datasets named `images/image{idx:09d}`, where `idx` is the image index. Each image dataset has the following attributes:
- `event_idx`: The index of the event corresponding to the image timestamp.
- `timestamp`: The timestamp of the image (`uint64`), aligned with the timestamps of the event stream.

## Model weight download

Inference of the code requires the following pretrained model weights:

1. [V2V-E2VID](https://github.com/HYLZ-2019/V2V): The weights are used for FRT & the review system. Please download `epoch_0077.pth` from [Google Drive](https://drive.google.com/file/d/1pCcu74dwQeYj8HI2TbOWkkAdnuoi9bW7/view?usp=drive_link) and move it to `adaptive_e2vid/checkpoints/v2v_e2vid_10k/epoch_0077.pth`.

2. Adaptive-E2VID: The weights are used for ART. Please download `epoch_0059.pth` from [Google Drive](https://drive.google.com/drive/folders/1jea3bJvSy89Gu1Y8bYAn-JkzQziRz7iA?usp=sharing) and move it to `adaptive_e2vid/checkpoints/adaptive_e2vid/epoch_0059.pth`.

3. Qwen3-VL: The weights are used for both FRT & ART. Models of any size (2B/4B/8B/32B/...) can be used (note that the paper uses the Thinking versions). They can be downloaded from [HuggingFace](https://huggingface.co/collections/Qwen/qwen3-vl) or [ModelScope](https://modelscope.cn/collections/Qwen3-VL-5c7a94c8cb144b). Let `{qwen_root}` be the directory to the downloaded local Qwen3-VL models, so that paths such as `{qwen_root}/Qwen3-VL-2B-Thinking/config.json` exist. This path will be required in the inference code.

## Environment setup

Required packages are listed in `requirements.txt`. The requirements are not strict: using different versions for most packages is fine.
- Qwen3-VL requires transformers>=4.57.0.
- I use torch==2.9.0. Other versions may also work.

## EvQA Benchmark Review

To review the questions from the EvQA dataset:

```bash
cd review_system
python server.py --release --dataset_root {evqa_root}
```

This will launch a local server at http://127.0.0.1:5000, in which you can review the questions. Videos are generated from the event data when visiting the corresponding question, so loading may take some time. If you want to pre-generate all videos, run:

```bash
python server.py --release --dataset_root {evqa_root} --make_cache
```

## Method inference

Edit `event_vl/configs.py` so that the content is:

```python
QWEN_ROOT = "{qwen_root}"
EVQA_ROOT = "{evqa_root}"
```

For inference, change to the `event_vl` directory.

```bash
cd event_vl
```

To quantitatively test on the EvQA benchmark, run:

```bash
python test_art.py
python test_frt.py
```

The answers from the models will be logged to `event_vl/results`, and metrics will be printed to the console.

To qualitatively test on open-ended questions, run:

```bash
python art_frt_free_inference.py
```

## Adaptive-E2VID training

To train the Adaptive-E2VID model from scratch, change to the `adaptive_e2vid` directory. The training framework is similar to that of [V2V-E2VID](https://github.com/HYLZ-2019/V2V), but with modifications for adaptive triggering. Prepare the WebVid training data and change the path in `config/webvid_root.txt` accordingly. Then, train with:

```bash
cd adaptive_e2vid
python train.py config/adaptive_e2vid.yaml
```

## Citation

If you find this repository useful for your research, please consider starring this repository ✨ and/or citing our paper 🧻:

```
@InProceedings{lou2025evqa,
  title={Reconstruction as a Bridge for Event-Based Visual Question Answering},
  author={Lou, Hanyue and Zhou, Jiayi and Zhang, Yang and Li, Boyu and Wang, Yi and Guangnan, Ye and Shi, Boxin},
  booktitle={arXiv preprint arXiv:2512.11510},
  year={2025}
}
```

Please feel free to open an issue if you meet with any problems or have any questions regarding the paper.