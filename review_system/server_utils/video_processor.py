"""
Video processing module for the annotation system.
Handles conversion of H5 event data to video files using FFmpeg.
"""

import h5py
import numpy as np
import cv2
import tempfile
import os
import shutil
import subprocess
import sys

def map_color(val, clip=10):
    """Map event values to colors for visualization"""
    # val.shape: (T, H, W)
    BLUE = np.array([255, 0, 0]).reshape((1, 1, 1, 3))
    RED = np.array([0, 0, 255]).reshape((1, 1, 1, 3))
    WHITE = np.array([255, 255, 255]).reshape((1, 1, 1, 3))
    val = np.clip(val, -clip, clip)
    val = np.expand_dims(val, -1) # (T, H, W, 1)
    red_side = (1 - val / clip) * WHITE + (val / clip) * RED
    blue_side = (1 + val / clip) * WHITE + (-val / clip) * BLUE
    return np.where(val > 0, red_side, blue_side).astype(np.uint8)

def h5_to_voxel(h5_path, bin_num):
    with h5py.File(h5_path, 'r') as f:                
        H , W = f.attrs['sensor_resolution']
        voxel = np.zeros((bin_num, H, W), dtype=np.int8)
        #print("Voxel created, voxel memory usage: ", voxel.nbytes / (1024*1024*1024), "GB")
        event_cnt = f["events/ts"].shape[0]
        t_begin = f["events/ts"][0]
        t_end = f["events/ts"][-1]
        t_per_bin = (t_end - t_begin + 1e-6) / bin_num # Avoid division by zero

        # Read out events, 1e6 events at a time
        EV_BATCH_SIZE = 1e8
        for start_idx in range(0, event_cnt, int(EV_BATCH_SIZE)):
            end_idx = min(start_idx + int(EV_BATCH_SIZE), event_cnt)
            #print(f"Processing events {start_idx} to {end_idx} / {event_cnt}")
            xs = f["events/xs"][start_idx:end_idx]
            ys = f["events/ys"][start_idx:end_idx]
            ts = f["events/ts"][start_idx:end_idx]
            ps = f["events/ps"][start_idx:end_idx].astype(np.int8)
            ps = np.where(ps > 0, 1, -1)  # Convert polarities to 1 and -1
            
            bin_idx = ((ts - t_begin) / t_per_bin).astype(np.int32)
            np.add.at(voxel, (bin_idx, ys, xs), ps)
            #print(f"Finished adding events {start_idx} to {end_idx}")

    #print("Finished adding events.")
    return voxel

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

def h5_to_videos(h5_path, vis_output_path=None, e2vid_output_path=None, reconstructor=None):
    """Convert .h5 event data to video file using FFmpeg"""

    FPS = 24
    with h5py.File(h5_path, 'r') as f:
        H, W = f.attrs["sensor_resolution"]   
        ts_dtype = f["events/ts"].dtype
        t_0, t_1 = f["events/ts"][0], f["events/ts"][-1]
        if ts_dtype not in [np.int64, np.uint64, np.int32, np.uint32]:
            t_0, t_1 = int(t_0 * 1e6), int(t_1 * 1e6)
        total_frame_cnt = int((t_1 - t_0) * FPS / 1e6) + 1  # Convert microseconds to seconds
        print("total_frame_cnt: ", total_frame_cnt)
        
        do_e2vid = e2vid_output_path is not None and reconstructor is not None
        do_vis = vis_output_path is not None

        if do_e2vid and do_vis:
            e2vid_voxels = h5_to_voxel(h5_path, 5 * total_frame_cnt).reshape((total_frame_cnt, 5, H, W))
            vis_voxels = e2vid_voxels.sum(axis=1)
        elif do_vis:
            vis_voxels = h5_to_voxel(h5_path, total_frame_cnt)
        elif do_e2vid:
            e2vid_voxels = h5_to_voxel(h5_path, 5 * total_frame_cnt).reshape((total_frame_cnt, 5, H, W))
        else:
            print("Nothing to be done.")
            return True
        
        if do_vis:
            vis = map_color(vis_voxels)
            success = np_to_video(vis, vis_output_path)
            assert success

        if do_e2vid:
            recon = reconstructor.reconstruct(e2vid_voxels)
            print("recon.shape: ", recon.shape)
            success = np_to_video(recon, e2vid_output_path)
            assert success
    
    return True
        
