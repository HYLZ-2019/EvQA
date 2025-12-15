import numpy as np
from typing import List, Tuple, Dict, Any
import math
import os
import cv2
import h5py
from numpy.lib.stride_tricks import sliding_window_view

def print_patch_metadata(patches):
    # Print (ey, ex, eh, ew, begin_t, end_t) for each patch
    for i, p in enumerate(patches):
        print(f"Patch {i}: ey={p['ey']}, ex={p['ex']}, eh={p['eh']}, ew={p['ew']}, begin_t={p['begin_t']}, end_t={p['end_t']}")
    print("Total patches:", len(patches))

# Largest-rectangle-first tiling (exact cover, disjoint rectangles)
def _largest_rectangle_in_binary(binary_mat: np.ndarray) -> tuple:
    """Return (r0, r1_ex, c0, c1_ex, area). If no 1s, area=0."""
    r_cnt, c_cnt = binary_mat.shape
    heights = np.zeros(c_cnt, dtype=np.int32)
    best_area = 0
    best = (0, 0, 0, 0)

    for r in range(r_cnt):
        row = binary_mat[r]
        # update heights
        heights = heights + 1
        heights[~row] = 0

        # largest rectangle in histogram for this row
        stack = []  # stack of (col_index)
        # iterate with sentinel
        for c in range(c_cnt + 1):
            cur_h = heights[c] if c < c_cnt else 0
            while stack and heights[stack[-1]] > cur_h:
                h = heights[stack.pop()]
                left = stack[-1] + 1 if stack else 0
                width = c - left
                area = h * width
                if area > best_area:
                    best_area = area
                    c0 = left
                    c1 = c
                    r1_ex = r + 1
                    r0 = r1_ex - h
                    best = (r0, r1_ex, c0, c1)
            if c < c_cnt:
                stack.append(c)

    return (*best, best_area)

def merge_patches(patches: List[Dict[str, Any]], padded_H: int, padded_W: int, patch_size: int) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    For 32*32 patches with the same end_t, merge them according to spatial locations to form larger rectangle patches.
    """

    if not patches:
        return [], padded_H, padded_W

    ps = patch_size
    rows = int(padded_H // ps) if ps > 0 else 0
    cols = int(padded_W // ps) if ps > 0 else 0

    # Group patches by end_t
    by_end: Dict[Any, List[Dict[str, Any]]] = {}
    for p in patches:
        by_end.setdefault(p["end_t"], []).append(p)

    merged_all: List[Dict[str, Any]] = []

    for end_t, group in by_end.items():
        # Build occupancy grid and a map from (r,c) -> patch
        occ = np.zeros((rows, cols), dtype=bool)
        cell2patch: Dict[Tuple[int, int], Dict[str, Any]] = {}

        for p in group:
            ey, ex = int(p["ey"]), int(p["ex"])  # grid-aligned
            assert ey % ps == 0 and ex % ps == 0 and p["eh"] == ps and p["ew"] == ps, "Patch not aligned to grid or wrong size"
            r = ey // ps
            c = ex // ps
            if 0 <= r < rows and 0 <= c < cols:
                occ[r, c] = True
                cell2patch[(r, c)] = p

        # Iteratively pick the largest rectangle of True cells and remove it
        while np.any(occ):
            r0, r1, c0, c1, area = _largest_rectangle_in_binary(occ)
            # Rectangle spans rows [r0, r1) and cols [c0, c1)
            h_cells = int(r1 - r0)
            w_cells = int(c1 - c0)
            eh = h_cells * ps
            ew = w_cells * ps
            ey = r0 * ps
            ex = c0 * ps

            # Compose merged voxel by pasting each small patch into its slot
            sample_np = cell2patch[(r0, c0)]["voxel"]
            num_bins = int(sample_np.shape[0])
            merged_voxel = np.zeros((num_bins, eh, ew), dtype=sample_np.dtype)

            begin_t_min = None
            for rr in range(h_cells):
                for cc in range(w_cells):
                    cell = (r0 + rr, c0 + cc)
                    p = cell2patch.get(cell)
                    if p is None:
                        # This can occur if occ had a True without stored patch (defensive)
                        continue
                    v = p["voxel"]
                    ys = rr * ps
                    ye = ys + ps
                    xs = cc * ps
                    xe = xs + ps
                    merged_voxel[:, ys:ye, xs:xe] = v

                    bt = p.get("begin_t", None)
                    if bt is not None:
                        begin_t_min = bt if begin_t_min is None else min(begin_t_min, bt)

            merged_patch = {
                "voxel": merged_voxel,
                "ey": ey,
                "ex": ex,
                "eh": eh,
                "ew": ew,
                "begin_t": begin_t_min if begin_t_min is not None else end_t,
                "end_t": end_t,
            }
            merged_all.append(merged_patch)

            # Remove the used area to keep rectangles disjoint and exact cover
            occ[r0:r1, c0:c1] = False

    return merged_all, padded_H, padded_W

class TriggerParamGenerator:
    def __init__(self, exp_name="default"):
        self.exp_name = exp_name
    
    def __call__(self, total_event_cnt, duration, H, W, patch_per_time):
        if self.exp_name == "fixed_threshold":
            patch_size = 32
            delta_t = int(1e6 / 4)
            event_batch_size = max(6000, total_event_cnt / (duration / delta_t) * patch_per_time)
            trigger_rate = 0.5 * patch_per_time
        
        else:
            assert False, f"Unknown experiment name: {self.exp_name}"

        return int(event_batch_size), int(trigger_rate), int(patch_size)
    

class VoxelPatchTrigger:
    def __init__(self, 
                 trigger_type: str = "naive",
                 num_bins: int = 5,
                 bin_type: str = "time",
                 patch_size: int = 64,
                 trigger_rate: float = 0.1,
                 patch_dim: int = 2,
                 patch_per_time: int = 2):
        """
        Initialize Voxel Patch Trigger class
        
        Args:
            trigger_type: Trigger strategy type, "naive" or "bias"
            num_bins: Number of bins for voxel
            bin_type: Bin type, "time" or "events"
            patch_size: Patch size
            trigger_rate: Trigger rate, event density required per patch
            patch_dim: Only for bias strategy, dimension of minipatch
            patch_pertime: Number of patches to trigger per time interval (1 or 2)
        """
        self.trigger_type = trigger_type
        self.num_bins = num_bins
        self.bin_type = bin_type
        self.patch_size = patch_size
        self.trigger_rate = trigger_rate
        self.patch_dim = patch_dim
        self.patch_per_time = patch_per_time
        
        # Validate parameters
        if trigger_type not in ["naive", "bias", "merge"]:
            raise ValueError(f"Unsupported trigger_type: {trigger_type}")
        if bin_type not in ["time", "events"]:
            raise ValueError(f"Unsupported bin_type: {bin_type}")
        if trigger_type == "bias" and patch_size % patch_dim != 0:
            raise ValueError("patch_size should be divisible by patch_dim for bias trigger")
    
    def rebin_voxel_time(self, voxel_slice: np.ndarray) -> np.ndarray:
        """Rebin voxel by simply splitting time dimension """
        if voxel_slice.shape[0] == 0:
            return np.zeros((self.num_bins, *voxel_slice.shape[1:]), dtype=np.float32)
        
        chunks = np.array_split(voxel_slice, self.num_bins, axis=0)
        return np.array([np.sum(chunk, axis=0) for chunk in chunks],dtype=np.float32)
    
    def rebin_voxel_events(self, voxel_slice: np.ndarray) -> np.ndarray:
        """Rebin voxel by event count evenly """
        T = voxel_slice.shape[0]
        if T == 0:
            return np.zeros((self.num_bins, *voxel_slice.shape[1:]), dtype=np.float32)
        # print("Rebinning voxel slice of shape:", voxel_slice.shape)
        sum_events= np.sum(np.abs(voxel_slice), axis=(1, 2))
        '''
        print("sum_events:", sum_events)
        print("Total events:", np.sum(np.abs(voxel_slice[0:1])))
        print("Total events:", np.sum(np.abs(voxel_slice[1:2])))
        '''
        cum_events = np.cumsum(sum_events)
        total_events = cum_events[-1]
        
        if total_events == 0:
            return np.zeros((self.num_bins, *voxel_slice.shape[1:]), dtype=np.float32)
        
        bin_edges = np.linspace(0, total_events, self.num_bins + 1)
        bin_indices = np.searchsorted(cum_events, bin_edges)
    
        '''
        print("bin_indices:", bin_indices)
        print("cum_events:", cum_events)
        print("bin_edges:", bin_edges)
        '''
        
        rebinned = np.zeros((self.num_bins, *voxel_slice.shape[1:]), dtype=np.float32)
        
        for i in range(self.num_bins):
            start, end = bin_indices[i], bin_indices[i+1]
            # print(start,end)
            if start < end:
                rebinned[i] = np.sum(voxel_slice[start:end], axis=0)
        
        return rebinned
    
    def find_max_patch(self, event_counts: np.ndarray, rows: int, cols: int, patch_dim: int) -> Tuple[int, int]:
        """Find the patch region with maximum accumulation"""
        windows = sliding_window_view(event_counts, (patch_dim, patch_dim))
        patch_sums = np.sum(windows, axis=(-2, -1))
        max_idx = np.argmax(patch_sums)
        max_r, max_c = np.unravel_index(max_idx, patch_sums.shape)
        
        return max_r, max_c, patch_sums[max_r, max_c]
    
    def reconstruct_time_slice_from_events_optimized(self, minipatch_data: List, minipatch_events_time: List, 
                                                r_start: int, r_end: int, c_start: int, c_end: int,end_time: int) -> Tuple[np.ndarray, int, int]:
        """Reconstruct complete time series from minipatch data using direct accumulation"""
        # Get patch dimensions
        minipatch_size = self.patch_size // self.patch_dim
        patch_height = (r_end - r_start) * minipatch_size
        patch_width = (c_end - c_start) * minipatch_size
        
        # Find time range
        start_time = 0
        
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                start_time = min(start_time, minipatch_events_time[r][c])
        
        if start_time == float('inf'):
            return np.array([]), 0, 0
        
        time_length = end_time - start_time + 1
        
        # Initialize the 3D array with zeros
        time_slices = np.zeros((time_length, patch_height, patch_width), dtype=np.int32)

        # Accumulate data directly into the pre-allocated array
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                if r < len(minipatch_data) and c < len(minipatch_data[0]) and minipatch_events_time[r][c] != -1:
                    # Calculate the spatial position of this minipatch in the final patch
                    y_start = (r - r_start) * minipatch_size
                    y_end = y_start + minipatch_size
                    x_start = (c - c_start) * minipatch_size
                    x_end = x_start + minipatch_size
                    
                    # Process each time frame in this minipatch
                    local_start_time = minipatch_events_time[r][c]
                    for t_local, data in enumerate(minipatch_data[r][c]):
                        t_global = local_start_time + t_local
                        if start_time <= t_global <= end_time:
                            time_slices[t_global-start_time, y_start:y_end, x_start:x_end] = data
            
        return time_slices, start_time
    
    def process_triggered_patch(self, patch_voxel , y_start, y_size , x_start, x_size, start_time, end_time) -> Dict[str, Any]:
        '''
        Process a triggered patch and apply binning
        Args:
            patch_voxel: [bin_time, H, W] numpy array of the patch voxel time series
            y_start: Starting y coordinate of the patch
            y_size: Height of the patch
            x_start: Starting x coordinate of the patch
            x_size: Width of the patch
            start_time: Start time of the patch
            end_time: End time of the patch
        Returns:
            patch: Dict with keys 'voxel', 'ey', 'ex', 'eh', 'ew', 'begin_t', 'end_t'
        '''
        time_slices = np.array(patch_voxel)
        #print("sum:",np.sum(np.abs(time_slices)))
        if self.bin_type == 'time':
            patch_voxel = self.rebin_voxel_time(time_slices)
        else:  # events
            patch_voxel = self.rebin_voxel_events(time_slices)
        
        patch = {
            'voxel': patch_voxel,
            'ey': y_start,
            'ex': x_start,
            'eh': y_size,
            'ew': x_size,
            'begin_t': start_time,
            'end_t': end_time,
        }
        return patch
    
    def split_patch(self,patch_voxel , y_start, y_size , x_start, x_size, start_time, end_time,all_triggered_patches):
        # print(self.bin_type,end_time-start_time+1)
        if (self.patch_per_time == 1):
            all_triggered_patches.append(
                self.process_triggered_patch(
                    patch_voxel, 
                    y_start, y_size , x_start, x_size, start_time, end_time,
                )
            )
        else:
            total_time = end_time - start_time + 1
            split_time = total_time // 2
            all_triggered_patches.append(
                self.process_triggered_patch(
                    patch_voxel[:split_time], 
                    y_start, y_size , x_start, x_size, start_time, start_time +split_time-1,
                )
            )
            all_triggered_patches.append(
                self.process_triggered_patch(
                    patch_voxel[split_time:], 
                    y_start, y_size , x_start, x_size, start_time +split_time-1, end_time,
                )
            )

    def naive_trigger_voxel(self, voxel: np.ndarray) -> Tuple[List, int, int]:
        """Naive trigger strategy - from frame voxel"""
        T, H, W = voxel.shape
        
        patch_size = self.patch_size
        patch_rows = int(np.ceil(H / patch_size))
        patch_cols = int(np.ceil(W / patch_size))
        # print("patch_rows, patch_cols:", patch_rows, patch_cols)
        padded_H = patch_rows * patch_size
        padded_W = patch_cols * patch_size
        
        pixels_per_patch = np.zeros((patch_rows, patch_cols), dtype=np.int32)
        for r in range(patch_rows):
            for c in range(patch_cols):
                rh = min(patch_size, H - r * patch_size)
                cw = min(patch_size, W - c * patch_size)
                pixels_per_patch[r, c] = rh * cw
        
        # Store event list for each patch - now storing (t, patch_data)
        patch_events_data = [[[] for _ in range(patch_cols)] for _ in range(patch_rows)]
        patch_events_time = [[0 for _ in range(patch_cols)] for _ in range(patch_rows)]
        # Store accumulation for each patch
        patch_accumulations = [[0 for _ in range(patch_cols)] for _ in range(patch_rows)]
        
        all_triggered_patches = []
        current_frame_pad = np.zeros((padded_H, padded_W), dtype=voxel.dtype)
        timestep = self.num_bins * self.patch_per_time

        for t in range(T):
            current_frame = voxel[t]
            current_frame_pad[:] = 0
            current_frame_pad[:H, :W] = current_frame
            
            # Update event list and accumulation for each patch
            for r in range(patch_rows):
                for c in range(patch_cols):
                    y_start = r * patch_size
                    y_end = y_start + patch_size
                    x_start = c * patch_size  
                    x_end = x_start + patch_size
                    
                    patch_data = current_frame_pad[y_start:y_end, x_start:x_end].copy()
                   
                    patch_events_data[r][c].append(patch_data)
                    patch_sum = np.sum(np.abs(patch_events_data[r][c][-1]))
                    # print(f"Time {t}, Patch ({r},{c}), Patch Sum: {patch_sum}")
                    patch_accumulations[r][c] += patch_sum

                    # print(f"Time {t}, Patch ({r},{c}), Sum: {patch_sum}, Accumulation: {patch_accumulations[r][c]}")
            
                    if t-patch_events_time[r][c]+1>=timestep and (patch_accumulations[r][c] >= pixels_per_patch[r, c] * self.trigger_rate or t == T-1 or t == timestep):
                        # 在最早的时刻全图触发一次，作为一个全图一致的初始化
                        # 在最末尾清仓剩下的event
                        self.split_patch(patch_events_data[r][c],y_start,patch_size,x_start,patch_size,patch_events_time[r][c],t,all_triggered_patches)
                        # reset
                        patch_events_time[r][c] = t+1
                        patch_events_data[r][c] = []
                        patch_accumulations[r][c] = 0
        
        return all_triggered_patches, padded_H, padded_W
    
    def bias_trigger_voxel(self, voxel: np.ndarray) -> Tuple[List, int, int]:
        """Optimized bias trigger strategy - from frame voxel"""
        T, H, W = voxel.shape
        patch_dim = self.patch_dim
        minipatch_size = self.patch_size // patch_dim
        
        minipatch_rows = int(np.ceil(H / minipatch_size))
        minipatch_cols = int(np.ceil(W / minipatch_size))

        padded_H = minipatch_rows * minipatch_size
        padded_W = minipatch_cols * minipatch_size
        
        # Store data list and start time for each minipatch
        minipatch_data = [[[] for _ in range(minipatch_cols)] for _ in range(minipatch_rows)]
        minipatch_events_time = [[0 for _ in range(minipatch_cols)] for _ in range(minipatch_rows)]
        # Store accumulation for each minipatch
        minipatch_accumulations = np.zeros((minipatch_rows, minipatch_cols), dtype=np.int32)
        
        all_triggered_patches = []
        
        for t in range(T):
            current_frame = voxel[t]
            current_frame_pad = np.zeros((padded_H, padded_W), dtype=current_frame.dtype)
            current_frame_pad[:H, :W] = current_frame

            # Update minipatch data list and accumulation
            for r in range(minipatch_rows):
                for c in range(minipatch_cols):
                    y_start = r * minipatch_size
                    y_end = y_start + minipatch_size
                    x_start = c * minipatch_size
                    x_end = x_start + minipatch_size

                    minipatch_data_frame = current_frame_pad[y_start:y_end, x_start:x_end]
                    minipatch_sum = np.sum(np.abs(minipatch_data_frame))
                    
                    minipatch_data[r][c].append(minipatch_data_frame)
                    minipatch_accumulations[r, c] += minipatch_sum

            # Find 2x2 minipatch region with maximum accumulation
            max_r, max_c, patch_big_sum = self.find_max_patch(minipatch_accumulations, minipatch_rows, minipatch_cols, 2*patch_dim)
            patch_big_pixels = patch_dim * patch_dim * minipatch_size * minipatch_size
            
            if t == T-1 or patch_big_sum >= patch_big_pixels * self.trigger_rate:
                # Trigger 2x2 region
                y_start = max_r * minipatch_size
                y_end = min((max_r + patch_dim*2) * minipatch_size, padded_H)
                x_start = max_c * minipatch_size
                x_end = min((max_c + patch_dim*2) * minipatch_size, padded_W)
                eh = y_end - y_start
                ew = x_end - x_start
                
                # Build time series from minipatch data
                time_slice, start_time = self.reconstruct_time_slice_from_events_optimized(
                    minipatch_data, minipatch_events_time, max_r, max_r+patch_dim*2, max_c, max_c+patch_dim*2, t
                )
                # print("Triggered big patch at time", t, "position (", y_start, x_start, ") size (", eh, ew, ")",patch_big_sum)
                assert np.sum(np.abs(time_slice)) == patch_big_sum, f"Reconstructed patch sum {np.sum(np.abs(time_slice))} does not match expected {patch_big_sum}"

                if time_slice.shape[0] > 0:
                    self.split_patch(patch_voxel=time_slice, y_start=y_start, y_size=eh, x_start=x_start, x_size=ew, 
                                    start_time=start_time, end_time=t, all_triggered_patches=all_triggered_patches)
                
                # Reset triggered minipatches
                for r in range(max_r, min(max_r+patch_dim*2, minipatch_rows)):
                    for c in range(max_c, min(max_c+patch_dim*2, minipatch_cols)):
                        minipatch_data[r][c] = []
                        minipatch_events_time[r][c] = t+1
                        minipatch_accumulations[r, c] = 0
            
            # Check other minipatch regions
            for r in range((max_r-1)%patch_dim+1-patch_dim, minipatch_rows, patch_dim):
                for c in range((max_c-1)%patch_dim+1-patch_dim, minipatch_cols, patch_dim):
                    realr = max(r,0)
                    realc = max(c,0)
                    y_start = realr * minipatch_size
                    dim_y = min(patch_dim, minipatch_rows - realr)
                    x_start = realc * minipatch_size
                    dim_x = min(patch_dim, minipatch_cols - realc)
                    # Skip recently triggered region
                    if max_r <= realr < max_r + patch_dim*2 and max_c <= realc < max_c + patch_dim*2:
                        continue
                    
                    patch_sum = np.sum(minipatch_accumulations[realr:realr+dim_y, realc:realc+dim_x])
                    patch_pixels = dim_x * dim_y * minipatch_size * minipatch_size
                    
                    if patch_sum >= patch_pixels * self.trigger_rate:
                        
                        # Build time series from minipatch data
                        time_slice, start_time = self.reconstruct_time_slice_from_events_optimized(
                            minipatch_data, minipatch_events_time, realr, realr+dim_y, realc, realc+dim_x, t
                        )
                        assert np.sum(np.abs(time_slice)) == patch_sum, f"Reconstructed patch sum {np.sum(np.abs(time_slice))} does not match expected {patch_sum}"
                        # print("Triggered small patch at time", t, "position (", y_start, x_start, ") size (", dim_y*minipatch_size, dim_x*minipatch_size, ")",patch_sum)
                        if time_slice.shape[0] > 0:
                            self.split_patch(patch_voxel=time_slice, y_start=y_start, y_size=dim_y*minipatch_size, 
                                            x_start=x_start, x_size=dim_x*minipatch_size, 
                                            start_time=start_time, end_time=t, 
                                            all_triggered_patches=all_triggered_patches)
                        
                        # Reset triggered minipatches
                        for rr in range(realr, realr+dim_y):
                            for cc in range(realc, realc+dim_x):
                                minipatch_data[rr][cc] = []
                                minipatch_events_time[rr][cc] = t+1
                                minipatch_accumulations[rr, cc] = 0
        
        return all_triggered_patches, padded_H, padded_W
    
    def process(self, voxel: np.ndarray) -> Tuple[List, int, int]:
        """
        Process voxel data and generate triggered patch sequence
        
        Args:
            voxel: Frame voxel with shape (T, H, W)
            
        Returns:
            all_triggered_patches: List of triggered patches, each element is 
                (voxel, ey, ex, eh, ew, start_time, end_time )
            padded_H, padded_W: Padded image dimensions
        """
        if len(voxel.shape) != 3:
            raise ValueError(f"Expected voxel with shape (T, H, W), got {voxel.shape}")
        
        if self.trigger_type == "naive":
            return self.naive_trigger_voxel(voxel)
        elif self.trigger_type == "bias":
            return self.bias_trigger_voxel(voxel)
        elif self.trigger_type == "merge":
            patches, padded_H, padded_W = self.naive_trigger_voxel(voxel)
            merged_patches, padded_H, padded_W = merge_patches(patches, padded_H, padded_W, self.patch_size)
            return merged_patches, padded_H, padded_W
        else:
            raise ValueError(f"Unsupported trigger_type: {self.trigger_type}")
        
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {
            'trigger_type': self.trigger_type,
            'num_bins': self.num_bins,
            'bin_type': self.bin_type,
            'patch_size': self.patch_size,
            'trigger_rate': self.trigger_rate,
            'patch_dim': self.patch_dim
        }


class EventPatchTrigger:
    def __init__(
            self, 
            trigger_type: str = "naive",
            num_bins: int = 5,
            bin_type: str = "time",
            patch_size: int = 64,
            trigger_rate: float = 0.1,
            patch_dim: int = 2,
            event_batch_size: int = 10000,
            patch_per_time: int = 2,
            trigger_param_gen_name: str = "e251106",
        ):
        """
        Initialize Event Patch Trigger class
        
        Args:
            trigger_type: Trigger strategy type, "naive" or "bias"
            num_bins: Number of bins for voxel
            bin_type: Bin type, "time" or "events"
            patch_size: Patch size
            trigger_rate: Trigger rate, event density required per patch
            patch_dim: Only for bias strategy, dimension of minipatch
            event_batch_size: Number of events to process in each batch
        """
        self.trigger_type = trigger_type
        self.num_bins = num_bins
        self.bin_type = bin_type
        #self.patch_size = patch_size
        #self.trigger_rate = trigger_rate
        self.patch_dim = patch_dim
        #self.event_batch_size = event_batch_size
        self.patch_per_time = patch_per_time
        # Validate parameters
        if trigger_type not in ["naive", "bias", "merge"]:
            raise ValueError(f"Unsupported trigger_type: {trigger_type}")
        if bin_type not in ["time", "events"]:
            raise ValueError(f"Unsupported bin_type: {bin_type}")
        if trigger_type == "bias" and patch_size % patch_dim != 0:
            raise ValueError("patch_size should be divisible by patch_dim for bias trigger")
        
        self.trigger_param_generator = TriggerParamGenerator(exp_name=trigger_param_gen_name)
    
    def find_max_patch(self, event_counts: np.ndarray, rows: int, cols: int, patch_dim: int) -> Tuple[int, int]:
        """Find the patch region with maximum accumulation"""
        
        windows = sliding_window_view(event_counts, (patch_dim, patch_dim))
        patch_sums = np.sum(windows, axis=(-2, -1))
        max_idx = np.argmax(patch_sums)
        max_r, max_c = np.unravel_index(max_idx, patch_sums.shape)
        
        return max_r, max_c
    
    def event_to_voxel_time(self, xs, ys, ts, ps, H, W):
        """Rebin voxel by time evenly"""
        voxel = np.zeros((self.num_bins, H, W), dtype=np.float32)
        if len(ts) == 0:
            return voxel
        
        t_per_bin = (ts[-1] - ts[0]) / self.num_bins + 1e-6  # avoid div zero
        bin_idx = ((ts - ts[0]) // t_per_bin).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)

        np.add.at(voxel, (bin_idx, ys, xs), ps)
        return voxel
    
    def event_to_voxel_event(self, xs, ys, ts, ps, H, W):
        """Rebin voxel by event count evenly"""
        N = xs.shape[0]
        voxel = np.zeros((self.num_bins, H, W), dtype=np.float32)
        
        if N == 0:
            return voxel
        
        ranks = np.arange(N, dtype=np.int64)
        bin_idx = ((ranks * self.num_bins) // max(N, 1)).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)

        np.add.at(voxel, (bin_idx, ys, xs), ps)
        return voxel
    
    def process_triggered_patch(self, x_start,x_size,y_start,y_size,xs, ys, ts,ps) -> Dict[str, Any]:
        '''
        Process a triggered patch and apply binning
        Args:
            x_start: Starting x coordinate of the patch
            x_size: Width of the patch
            y_start: Starting y coordinate of the patch
            y_size: Height of the patch
            xs, ys, ts, ps: Events within the patch
        Returns:d
            patch: Dict with keys 'voxel', 'ey', 'ex', 'eh', 'ew', 'begin_t', 'end_t'
        '''
        #print("sum:",np.sum(np.abs(time_slices)))
        if self.bin_type == 'time':
            patch_voxel = self.event_to_voxel_time(xs, ys, ts,ps,y_size,x_size)
        else:  # events
            patch_voxel = self.event_to_voxel_event(xs, ys, ts,ps,y_size,x_size)
        
        patch = {
            'voxel': patch_voxel,
            'ey': y_start,
            'ex': x_start,
            'eh': y_size,
            'ew': x_size,
            'begin_t': ts[0],
            'end_t': ts[-1],
        }
        return patch
    
    def split_patch(self, xs, ys, ts, ps, y_start, y_size, x_start, x_size, all_triggered_patches):
        """
        Split events by midtime and create patches for each half
        Args:
            xs, ys, ts, ps: Events in the patch
            y_start, y_size: Patch y position and height
            x_start, x_size: Patch x position and width
            all_triggered_patches: List to append generated patches to
        """
        if len(ts) == 0:
            return
        
        if self.patch_per_time == 1:
            # No splitting, create single patch
            patch = self.process_triggered_patch(x_start, x_size, y_start, y_size, 
                                            xs, ys, ts, ps)
            all_triggered_patches.append(patch)
            return
        
        # Calculate midtime (midpoint between first and last event timestamp)
        begin_t = ts[0]
        end_t = ts[-1]
        midtime = (begin_t + end_t) / 2
        
        # Split events by midtime
        first_half_mask = ts <= midtime
        second_half_mask = ts > midtime
        
        # Create patch for first half (earlier events)
        if np.any(first_half_mask):
            xs_first = xs[first_half_mask]
            ys_first = ys[first_half_mask]
            ts_first = ts[first_half_mask]
            ps_first = ps[first_half_mask]
            
            patch_first = self.process_triggered_patch(x_start, x_size, y_start, y_size,
                                                    xs_first, ys_first, ts_first, ps_first)
            all_triggered_patches.append(patch_first)
        
    # Create patch for second half (later events)
        if np.any(second_half_mask):
            xs_second = xs[second_half_mask]
            ys_second = ys[second_half_mask]
            ts_second = ts[second_half_mask]
            ps_second = ps[second_half_mask]
            
            patch_second = self.process_triggered_patch(x_start, x_size, y_start, y_size,
                                                    xs_second, ys_second, ts_second, ps_second)
            all_triggered_patches.append(patch_second)

    def naive_trigger_events(self, h5_path: str) -> Tuple[List, int, int]:
        """Naive trigger strategy for events (optimized version)"""
        h5_file = h5py.File(h5_path, 'r')
        H, W = h5_file.attrs['sensor_resolution']
        total_event_cnt = h5_file['events/xs'].shape[0]
        duration = h5_file['events/ts'][total_event_cnt - 1] - h5_file['events/ts'][0]

        self.event_batch_size, self.trigger_rate, patch_size = self.trigger_param_generator(
            total_event_cnt, duration, H, W, self.patch_per_time
        )

        patch_rows = np.ceil(H / patch_size).astype(int)
        patch_cols = np.ceil(W / patch_size).astype(int)

        patch_heights = np.array([min(patch_size, H - r * patch_size) for r in range(patch_rows)])
        patch_widths = np.array([min(patch_size, W - c * patch_size) for c in range(patch_cols)])
        pixels_per_patch = patch_heights[:, None] * patch_widths[None, :]
        
        # Memory-efficient ring buffers per patch using compact dtypes
        # Use user's precise buffer size formula (guaranteed sufficient)
        BUFFER_SIZE = int(self.trigger_rate * pixels_per_patch.max() + self.event_batch_size)
        BUFFER_SIZE = int(min(BUFFER_SIZE, 4e6))
        
        dtype_xy = np.uint8 if patch_size <= 256 else np.uint16
        if patch_size > 256:
            print(f"[INFO] patch_size={patch_size} > 256, using uint16 for local coords to avoid overflow")
        patch_xs = np.zeros((patch_rows, patch_cols, BUFFER_SIZE), dtype=dtype_xy)
        patch_ys = np.zeros((patch_rows, patch_cols, BUFFER_SIZE), dtype=dtype_xy)
        patch_ts = np.zeros((patch_rows, patch_cols, BUFFER_SIZE), dtype=np.uint32)  # relative to base per patch
        patch_ps = np.zeros((patch_rows, patch_cols, BUFFER_SIZE), dtype=np.int8)    # -1/+1
        patch_t_base = np.zeros((patch_rows, patch_cols), dtype=np.int64)            # absolute base ts per patch
        patch_event_counts = np.zeros((patch_rows, patch_cols), dtype=np.int32)      # valid count per patch

        all_triggered_patches = []
        #print(f"H,W:{(H, W)} Adjusted patch_size: {patch_size}, event_batch_size: {self.event_batch_size}, trigger_rate: {self.trigger_rate:.4f}, buffer={BUFFER_SIZE}")

        for i in range(0, total_event_cnt, self.event_batch_size):
            xs = h5_file['events/xs'][i:i + self.event_batch_size]
            ys = h5_file['events/ys'][i:i + self.event_batch_size]
            ts = h5_file['events/ts'][i:i + self.event_batch_size]
            ps = h5_file['events/ps'][i:i + self.event_batch_size].astype(np.int8)
            ps = np.where(ps > 0, 1, -1)

            if xs.size == 0:
                continue

            patch_rs = ys // patch_size
            patch_cs = xs // patch_size
            flat = patch_rs * patch_cols + patch_cs

            order = np.lexsort((ts, flat))
            flat_sorted = flat[order]
            events_sorted = np.stack([xs, ys, ts, ps], axis=1).astype(np.int64)[order]

            M = patch_rows * patch_cols
            batch_counts = np.bincount(flat_sorted, minlength=M)
            offsets = np.concatenate(([0], np.cumsum(batch_counts)))

            prev_counts = patch_event_counts.reshape(-1).copy()

            for idx_bin in np.nonzero(batch_counts)[0]:
                s, e = offsets[idx_bin], offsets[idx_bin + 1]
                if s == e:
                    continue
                r_idx, c_idx = divmod(idx_bin, patch_cols)
                prev = int(prev_counts[idx_bin])
                need = int(e - s)
                # Establish base ts for patch when starting a new buffer
                if prev == 0:
                    patch_t_base[r_idx, c_idx] = int(events_sorted[s, 2])
                base = patch_t_base[r_idx, c_idx]

                # Prepare local coordinates and relative timestamps
                xs_seg_abs = events_sorted[s:e, 0]
                ys_seg_abs = events_sorted[s:e, 1]
                ts_seg_abs = events_sorted[s:e, 2]
                ps_seg = events_sorted[s:e, 3].astype(np.int8)

                xs_seg = (xs_seg_abs - c_idx * patch_size).astype(dtype_xy, copy=False)
                ys_seg = (ys_seg_abs - r_idx * patch_size).astype(dtype_xy, copy=False)
                # Ensure non-negative deltas
                ts_rel = (ts_seg_abs - base).astype(np.int64)
                # Clip to uint32 range if necessary
                if ts_rel.max(initial=0) > np.iinfo(np.uint32).max:
                    print(f"[WARN] ts_rel overflow in patch (r={r_idx},c={c_idx}); clipping to uint32 max")
                ts_seg = np.clip(ts_rel, 0, np.iinfo(np.uint32).max).astype(np.uint32)

                if prev + need > BUFFER_SIZE:
                    # Fallback: shrink slice to fit and mark overflow
                    fit = BUFFER_SIZE - prev
                    print(f"[WARN] Truncating events for patch (r={r_idx},c={c_idx}) adding {fit}/{need}")
                    patch_xs[r_idx, c_idx, prev:prev + fit] = xs_seg[:fit]
                    patch_ys[r_idx, c_idx, prev:prev + fit] = ys_seg[:fit]
                    patch_ts[r_idx, c_idx, prev:prev + fit] = ts_seg[:fit]
                    patch_ps[r_idx, c_idx, prev:prev + fit] = ps_seg[:fit]
                    patch_event_counts[r_idx, c_idx] += fit
                    continue

                patch_xs[r_idx, c_idx, prev:prev + need] = xs_seg
                patch_ys[r_idx, c_idx, prev:prev + need] = ys_seg
                patch_ts[r_idx, c_idx, prev:prev + need] = ts_seg
                patch_ps[r_idx, c_idx, prev:prev + need] = ps_seg
                patch_event_counts[r_idx, c_idx] += need

            triggered_mask = patch_event_counts > (pixels_per_patch * self.trigger_rate).astype(np.int32)
            if not triggered_mask.any():
                continue
            triggered_indices = np.where(triggered_mask)
            for r, c in zip(triggered_indices[0], triggered_indices[1]):
                k = int(patch_event_counts[r, c])
                if k == 0:
                    continue
                xs_local = patch_xs[r, c, :k]
                ys_local = patch_ys[r, c, :k]
                ts_abs = patch_ts[r, c, :k].astype(np.uint64) + np.uint64(patch_t_base[r, c])
                ps_b = patch_ps[r, c, :k]
                self.split_patch(xs=xs_local, ys=ys_local, ts=ts_abs, ps=ps_b,
                                 y_start=r * patch_size, y_size=patch_size,
                                 x_start=c * patch_size, x_size=patch_size,
                                 all_triggered_patches=all_triggered_patches)
                patch_event_counts[r, c] = 0   # reset count; buffer reused
                patch_t_base[r, c] = 0         # clear base; next write will set new base

        #print(f"P2")
        h5_file.close()
        padded_H = patch_rows * patch_size
        padded_W = patch_cols * patch_size
        #print((H//32)*(H//32)*duration//1000000*8*0.2)
        #print("successfully triggered patches:", patch_size//32*patch_size//32*len(all_triggered_patches))
        return all_triggered_patches, padded_H, padded_W, patch_size
    
    def bias_trigger_events(self, h5_path: str) -> Tuple[List, int, int]:
        """Bias trigger strategy for events"""
        h5_file = h5py.File(h5_path, 'r')
        H, W = h5_file.attrs['sensor_resolution']
        total_event_cnt = h5_file['events/xs'].shape[0]
        duration = h5_file['events/ts'][total_event_cnt - 1] - h5_file['events/ts'][0]
        patch_size = math.ceil(H/10/32)*32  # adjust patch size according to image height   
        #patch_size = 32
        self.event_batch_size = max(6000,total_event_cnt // (duration  // 125000 )*self.patch_per_time)  # process events in 1 second batches
        self.trigger_rate = max(0.2*self.patch_per_time,self.event_batch_size/H/W*3*self.patch_per_time)  # adjust trigger rate according to event density
        patch_dim = self.patch_dim

        minipatch_size = patch_size // patch_dim
        assert patch_size % patch_dim == 0, "patch_size should be divisible by patch_dim"
        print (f"H,W:{H,W}Adjusted patch_size: {patch_size}, event_batch_size: {self.event_batch_size}, trigger_rate: {self.trigger_rate:.4f}")

        minipatch_rows = int(np.ceil(H / minipatch_size))
        minipatch_cols = int(np.ceil(W / minipatch_size))
        
        minipatch_heights = np.array([min(minipatch_size, H - r * minipatch_size) for r in range(minipatch_rows)])
        minipatch_widths = np.array([min(minipatch_size, W - c * minipatch_size) for c in range(minipatch_cols)])
        minipixels_per_patch = minipatch_heights[:, None] * minipatch_widths[None, :]

        minipatch_events = [[[] for _ in range(minipatch_cols)] for _ in range(minipatch_rows)]
        minipatch_event_counts = np.zeros((minipatch_rows, minipatch_cols), dtype=np.int32)
        all_triggered_patches = []

        for i in range(0, total_event_cnt, self.event_batch_size):
            xs = h5_file['events/xs'][i:i+self.event_batch_size]
            ys = h5_file['events/ys'][i:i+self.event_batch_size]
            ts = h5_file['events/ts'][i:i+self.event_batch_size]
            ps = h5_file['events/ps'][i:i+self.event_batch_size]
            ps = [1 if p > 0 else -1 for p in ps]  # Convert to +1/-1 polarity
            
            minipatch_rs = ys // minipatch_size
            minipatch_cs = xs // minipatch_size
            if len(minipatch_rs) == 0:
                continue
            
            # Update minipatch_event_counts
            flat_indices = minipatch_rs * minipatch_cols + minipatch_cs
            new_counts = np.bincount(flat_indices, minlength=minipatch_rows * minipatch_cols)
            minipatch_event_counts += new_counts.reshape(minipatch_rows, minipatch_cols)

            # Add events to minipatch lists
            for x, y, t, p in zip(xs, ys, ts, ps):
                r = y // minipatch_size
                c = x // minipatch_size
                minipatch_events[r][c].append((x, y, t, p))

            # Find max patch region
            max_r, max_c = self.find_max_patch(minipatch_event_counts, minipatch_rows, minipatch_cols, self.patch_dim * 2)
            
            # Check if 2x2 minipatch region meets trigger condition
            max_patch_events = []
            for r in range(max_r, min(max_r + patch_dim * 2, minipatch_rows)):
                for c in range(max_c, min(max_c + patch_dim * 2, minipatch_cols)):
                    max_patch_events.extend(minipatch_events[r][c])
            
            if len(max_patch_events) > 0:
                max_event = np.array(max_patch_events)
                max_event = max_event[np.argsort(max_event[:, 2])]  # Sort by timestamp
                
                ey = max_r * minipatch_size
                ex = max_c * minipatch_size
                eh = patch_size * 2
                ew = patch_size * 2
                
                xs, ys, ts, ps = max_event[:, 0], max_event[:, 1], max_event[:, 2], max_event[:, 3]
                xs = (xs - max_c * minipatch_size).astype(np.uint64)
                ys = (ys - max_r * minipatch_size).astype(np.uint64)

                self.split_patch(xs=xs, ys=ys, ts=ts, ps=ps, y_start=ey, y_size=eh,
                                 x_start=ex, x_size=ew, all_triggered_patches=all_triggered_patches)
                
                # Reset triggered minipatches
                for r in range(max_r, min(max_r + self.patch_dim * 2, minipatch_rows)):
                    for c in range(max_c, min(max_c + self.patch_dim * 2, minipatch_cols)):
                        minipatch_events[r][c] = []
                        minipatch_event_counts[r, c] = 0

            # Check other patch regions
            for r in range((max_r-1)%patch_dim+1-patch_dim, minipatch_rows, patch_dim):
                for c in range((max_c-1)%patch_dim+1-patch_dim, minipatch_cols, patch_dim):
                    realr = max(r,0)
                    realc = max(c,0)
                    ey = realr * minipatch_size
                    dim_y = min(patch_dim, minipatch_rows - realr)
                    ex = realc * minipatch_size
                    dim_x = min(patch_dim, minipatch_cols - realc)
                    # Skip recently triggered region
                    if max_r <= realr < max_r + patch_dim*2 and max_c <= realc < max_c + patch_dim*2:
                        continue
                    # Skip recently triggered region
                    
                    patch_sum = np.sum(minipatch_event_counts[realr:realr+dim_y, realc:realc+dim_x])
                    patch_pixels = dim_x * dim_y * minipatch_size * minipatch_size
                    
                    if patch_sum >= patch_pixels * self.trigger_rate:
                        # Collect events from this patch region
                        patch_events = []
                        for rr in range(r, min(r + self.patch_dim, minipatch_rows)):
                            for cc in range(c, min(c + self.patch_dim, minipatch_cols)):
                                patch_events.extend(minipatch_events[rr][cc])
                        
                        if len(patch_events) == 0:
                            continue
                            
                        events = np.array(patch_events)
                        events = events[np.argsort(events[:, 2])]  # Sort by timestamp
                        
                        xs, ys, ts, ps = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
                        xs = (xs - ex).astype(np.uint64)
                        ys = (ys - ey).astype(np.uint64)
                        
                        self.split_patch(xs=xs, ys=ys, ts=ts, ps=ps, y_start=ey, y_size=dim_y*minipatch_size,
                                            x_start=ex, x_size=dim_x*minipatch_size, all_triggered_patches=all_triggered_patches)                       
                        # Reset this patch region
                        for rr in range(realr, realr+dim_y):
                            for cc in range(realc, realc+dim_x):
                                minipatch_events[rr][cc] = []
                                minipatch_event_counts[rr, cc] = 0
        
        h5_file.close()
        padded_H = minipatch_rows * minipatch_size
        padded_W = minipatch_cols * minipatch_size
        
        return all_triggered_patches, padded_H, padded_W
    
    def process(self, h5_path: str) -> Tuple[List, int, int]:
        """
        Process event data from H5 file and generate triggered patch sequence
        
        Args:
            h5_path: Path to H5 file containing event data
            
        Returns:
            all_triggered_patches: List of triggered patches, each element is a dict with:
                'voxel': voxel tensor (num_bins, H, W)
                'ey', 'ex', 'eh', 'ew': patch coordinates and size
                'begin_t', 'end_t': time range of events in this patch
            padded_H, padded_W: Padded image dimensions
        """
      
        if self.trigger_type == "naive":
            patches, padded_H, padded_W, patch_size = self.naive_trigger_events(h5_path)
            return patches, padded_H, padded_W
        elif self.trigger_type == "bias":
            return self.bias_trigger_events(h5_path)
        elif self.trigger_type == "merge":
            patches, padded_H, padded_W, patch_size = self.naive_trigger_events(h5_path)
            #print("before merge: ", len(patches))
            merged_patches, padded_H, padded_W = merge_patches(patches, padded_H, padded_W, patch_size)
            #print("after merge:", len(merged_patches))
            return merged_patches, padded_H, padded_W
        else:
            raise ValueError(f"Unsupported trigger_type: {self.trigger_type}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {
            'trigger_type': self.trigger_type,
            'num_bins': self.num_bins,
            'bin_type': self.bin_type,
            'patch_size': self.patch_size,
            'trigger_rate': self.trigger_rate,
            'patch_dim': self.patch_dim,
            'event_batch_size': self.event_batch_size
        }

def _to_uint8(img: np.ndarray) -> np.uint8:
    img = img.astype(np.float32)
    m, M = np.min(img), np.max(img)
    if M > m:
        img = (img - m) / (M - m)
    else:
        img = np.zeros_like(img)
    return (img * 255.0).astype(np.uint8)

def visualize_patches_gray_two_panel(voxel: np.ndarray, patches: List[dict], fps: int = 30,
                                     out_path: str = "patch_two_panel.mp4") -> str:
    """
    生成灰度两联画视频：左=原始帧，右=触发的patch（只在 end_t 对应帧更新右侧画面）。
    - voxel: (T, H, W) 的帧事件图
    - patches: [{'ey','ex','eh','ew','voxel','end_t'}, ...]，voxel 为 (num_bins, eh, ew)
    """
    T, H, W = voxel.shape
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W * 2, H))

    # 预分组：每帧有哪些补丁（按 end_t）
    patches_by_t: Dict[int, List[dict]] = {}
    for p in patches:
        t = int(np.clip(np.rint(float(p["end_t"])), 0, T - 1))
        patches_by_t.setdefault(t, []).append(p)

    # 右侧画面缓冲（仅在触发帧更新）
    right_panel = np.zeros((H, W), dtype=np.uint8)

    for t in range(T):
        # 左：原始帧（每帧独立归一化到 uint8）
        left_gray = _to_uint8(voxel[t])
        #print(voxel[t])
        # 如果该帧有补丁触发，更新右侧画面（把每个 patch 的热度图贴回对应 ROI）
        if t in patches_by_t:
            # 先从上一帧继承（也可以清零看自己需求）
            # right_panel[:] = 0  # 若想每次只显示当前帧的补丁，可取消注释清零
            for p in patches_by_t[t]:
                ey, ex = int(p["ey"]), int(p["ex"])
                eh, ew = int(p["eh"]), int(p["ew"])

                ev = p["voxel"]  # (num_bins, eh, ew)
                if not isinstance(ev, np.ndarray):
                    try:
                        import torch
                        if isinstance(ev, torch.Tensor):
                            ev = ev.detach().cpu().numpy()
                        else:
                            ev = np.asarray(ev)
                    except Exception:
                        ev = np.asarray(ev)

                ev_sum = np.sum(np.abs(ev), axis=0)  # (eh, ew)
                #print(ev_sum)
                patch_u8 = _to_uint8((ev_sum))

                # 贴回 ROI（边界裁剪）
                y1, y2 = max(0, ey), min(ey + eh, H)
                x1, x2 = max(0, ex), min(ex + ew, W)
                ph, pw = y2 - y1, x2 - x1
                if ph > 0 and pw > 0:
                    right_panel[y1:y2, x1:x2] = patch_u8[:ph, :pw]
                

        # 拼接两联画，写入视频（OpenCV 常用 BGR 三通道）
        left_bgr = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
        right_bgr = cv2.cvtColor(right_panel, cv2.COLOR_GRAY2BGR)
        frame = np.concatenate([left_bgr, right_bgr], axis=1)
        cv2.putText(frame, f"t={t}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    return out_path

# Usage example
if __name__ == "__main__":
    # Create instances with different strategies
    naive_trigger = VoxelPatchTrigger(
        trigger_type="naive",
        num_bins=5,
        bin_type="time",
        patch_size=32,
        trigger_rate=0.3,
        patch_per_time=1
    )
    
    bias_trigger = VoxelPatchTrigger(
        trigger_type="bias",
        num_bins=5,
        bin_type="time",
        patch_size=32,
        trigger_rate=0.3,
        patch_dim=2,
        patch_per_time = 1,
    )
    '''
    # Simulate frame voxel (T=100, H=64, W=64)
    T, H, W = 20, 12, 12
    voxel = (np.random.rand(T, H, W) < 0.1).astype(np.int32)
    for t in range(T):
        print (f"Frame {t} has {np.sum(voxel[t])} events")
        print (voxel[t])
    # Process voxel data'''
    npz = np.load("/home/zjy/EventVL/sparse_e2vid/debug_output/vox_401_541244.npz"); vox = npz["voxels"]
    patches_naive, padded_H_naive, padded_W_naive = naive_trigger.process(vox)
    patches_bias, padded_H_bias, padded_W_bias = bias_trigger.process(vox)
    
    print(f"Naive strategy triggered {len(patches_naive)} patches")
    print(f"Bias strategy triggered {len(patches_bias)} patches")
    
    # Check configurations
    print("Naive config:", naive_trigger.get_config())
    print("Bias config:", bias_trigger.get_config())
    
    # Check triggered patches with time information
    '''
    for i, patch in enumerate(patches_bias):
        # print(patch)
        voxel_data = patch['voxel']
        ey = patch['ey']
        ex = patch['ex']    
        eh = patch['eh']
        ew = patch['ew']
        start_time = patch['begin_t']
        end_time = patch['end_t']
        print(f"Naive Patch {i}: position({ey},{ex}), size({eh}x{ew}), time_range({start_time}-{end_time})")
        print(f"  Voxel shape: {voxel_data.shape}")
    '''
    out1 = visualize_patches_gray_two_panel(vox, patches_naive, fps=30, out_path="patch_time.mp4")
    out2 = visualize_patches_gray_two_panel(vox, patches_bias, fps=30, out_path="patch_bias.mp4")
    print("Saved:", out1, out2)