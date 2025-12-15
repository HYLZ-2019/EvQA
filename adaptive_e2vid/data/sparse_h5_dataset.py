import torch
import h5py
import numpy as np

def event_to_voxel(xs, ys, ts, ps, num_bins, H, W):
    voxel = np.zeros((num_bins, H, W), dtype=np.float32)
    t_per_bin = (ts[-1] - ts[0]) / num_bins + 1e-6 # Avoid division by zero
    
    bin_idx = ((ts - ts[0]) // t_per_bin).astype(np.int32)

    np.add.at(voxel, (bin_idx, ys, xs), ps)
    return voxel

def event_to_voxel_events(xs, ys, ts, ps, num_bins, H, W):
    N = xs.shape[0]
    voxel = np.zeros((num_bins, H, W), dtype=np.float32)
    
    ranks = np.arange(N, dtype=np.int64)
    bin_idx = ((ranks * num_bins) // max(N, 1)).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    np.add.at(voxel,(bin_idx, ys, xs),ps)
    return voxel


def naive_read_sparse(h5_path, configs):
    num_bins = configs.get('num_bins', 5)
    patch_size = configs.get('patch_size', 64)
    trigger_rate = configs.get('trigger_rate', 0.1) # 0.1 means a 64*64 patch should have a sum of over 64*64*0.1 events to trigger
    event_batch_size = configs.get('event_batch_size', 10000) # Number of events to add before each trigger check

    h5_file = h5py.File(h5_path, 'r')
    H, W = h5_file.attrs['sensor_resolution']
    total_event_cnt = h5_file['events/xs'].shape[0]

    patch_rows = np.ceil(H / patch_size).astype(int)
    patch_cols = np.ceil(W / patch_size).astype(int)
    
    pixels_per_patch = np.zeros((patch_rows, patch_cols), dtype=np.int32)
    for r in range(patch_rows):
        for c in range(patch_cols):
            rh = min(patch_size, H - r * patch_size)
            cw = min(patch_size, W - c * patch_size)
            pixels_per_patch[r, c] = rh * cw
    
    patch_events = [[[] for _ in range(patch_cols)] for _ in range(patch_rows)]

    all_triggered_patches = []
    # Each element is (voxel, ey, ex, eh, ew). eh & ew are both patch_size (voxels of boundary patches are zero padded)
    for i in range(0, total_event_cnt, event_batch_size):
        xs = h5_file['events/xs'][i:i+event_batch_size]
        ys = h5_file['events/ys'][i:i+event_batch_size]
        ts = h5_file['events/ts'][i:i+event_batch_size]
        ps = h5_file['events/ps'][i:i+event_batch_size]
        for x, y, t, p in zip(xs, ys, ts, ps):
            r = y // patch_size
            c = x // patch_size
            patch_events[r][c].append((x, y, t, p))

        # Check triggers
        for r in range(patch_rows):
            for c in range(patch_cols):
                if len(patch_events[r][c]) >= pixels_per_patch[r, c] * trigger_rate:
                    # Triggered
                    events = np.array(patch_events[r][c])
                    if events.shape[0] == 0:
                        continue
                    xs, ys, ts, ps = events[:,0], events[:,1], events[:,2], events[:,3]
                    xs = xs - c * patch_size
                    ys = ys - r * patch_size
                    voxel = event_to_voxel(xs, ys, ts, ps, num_bins, patch_size, patch_size)
                    ey = r * patch_size
                    ex = c * patch_size
                    eh = patch_size
                    ew = patch_size
                    all_triggered_patches.append((voxel, ey, ex, eh, ew, ts[0], ts[-1]))
                    patch_events[r][c] = []
                    
        
    h5_file.close()


    padded_H = patch_rows * patch_size
    padded_W = patch_cols * patch_size

    return all_triggered_patches, padded_H, padded_W

def fast_read_sparse(h5_path, configs):
    num_bins = configs.get('num_bins', 5)
    patch_size = configs.get('patch_size', 64)
    trigger_rate = configs.get('trigger_rate', 0.1) # 0.1 means a 64*64 patch should have a sum of over 64*64*0.1 events to trigger
    event_batch_size = configs.get('event_batch_size', 10000) # Number of events to add before each trigger check

    h5_file = h5py.File(h5_path, 'r')
    H, W = h5_file.attrs['sensor_resolution']
    total_event_cnt = h5_file['events/xs'].shape[0]

    patch_rows = np.ceil(H / patch_size).astype(int)
    patch_cols = np.ceil(W / patch_size).astype(int)

    patch_heights = np.array([min(patch_size, H - r * patch_size) for r in range(patch_rows)])
    patch_widths = np.array([min(patch_size, W - c * patch_size) for c in range(patch_cols)])
    pixels_per_patch = patch_heights[:, None] * patch_widths[None, :]
    
    
    patch_events = [[[] for _ in range(patch_cols)] for _ in range(patch_rows)]
    patch_event_counts = np.zeros((patch_rows, patch_cols), dtype=np.int32)
    
    all_triggered_patches = []

    for i in range(0, total_event_cnt, event_batch_size):
        # fetch a batch of events
        xs = h5_file['events/xs'][i:i+event_batch_size]
        ys = h5_file['events/ys'][i:i+event_batch_size]
        ts = h5_file['events/ts'][i:i+event_batch_size]
        ps = h5_file['events/ps'][i:i+event_batch_size]
        
        patch_rs = ys // patch_size
        patch_cs = xs // patch_size
        
        if len(patch_rs) == 0:
            continue
        # update patch_event_counts
        flat_indices = patch_rs * patch_cols + patch_cs
        new_counts = np.bincount(flat_indices, minlength=patch_rows * patch_cols)
        patch_event_counts += new_counts.reshape(patch_rows, patch_cols)
        triggered_mask = patch_event_counts > (pixels_per_patch * trigger_rate).astype(np.int32)
        triggered_indices = np.where(triggered_mask)

        for x, y, t, p in zip(xs, ys, ts, ps):
            r = y // patch_size
            c = x // patch_size
            patch_events[r][c].append((x, y, t, p))
        
        for r, c in zip(triggered_indices[0], triggered_indices[1]):
            events = np.array(patch_events[r][c])
            if events.shape[0] == 0:
                continue
            xs, ys, ts, ps = events[:,0], events[:,1], events[:,2], events[:,3]
            xs = (xs - c * patch_size).astype(np.uint64)
            ys = (ys - r * patch_size).astype(np.uint64)
            voxel = event_to_voxel_events(xs, ys, ts, ps, num_bins, patch_size, patch_size)
            ey = r * patch_size
            ex = c * patch_size
            eh = patch_size
            ew = patch_size
            print(f"Triggered patch at ({ey},{ex}) size ({eh}x{ew}) time range ({ts[0]}-{ts[-1]})")
            all_triggered_patches.append((voxel, ey, ex, eh, ew,ts[0], ts[-1]))
            patch_events[r][c] = []
            patch_event_counts[r, c] = 0
        
 
    h5_file.close()


    padded_H = patch_rows * patch_size
    padded_W = patch_cols * patch_size

    return all_triggered_patches, padded_H, padded_W

def find_max_patch(event_counts, rows, cols, patch_dim):
    max_count = -1
    max_r, max_c = 0, 0
    for r in range(rows - patch_dim + 1):
        for c in range(cols - patch_dim + 1):
            current_sum = np.sum(event_counts[r:r+patch_dim, c:c+patch_dim])
            if current_sum > max_count:
                max_count = current_sum
                max_r, max_c = r, c
    return max_r, max_c

def bias_read_sparse(h5_path, configs):
    # patch_size should be divisible by patch_dim
    num_bins = configs.get('num_bins', 5)
    patch_size = configs.get('patch_size', 64)
    trigger_rate = configs.get('trigger_rate', 0.1) # 0.1 means a 64*64 patch should have a sum of over 64*64*0.1 events to trigger
    event_batch_size = configs.get('event_batch_size', 10000) # Number of events to add before each trigger check
    patch_dim = configs.get('patch_dim', 4) # Each patch is 4x4 minipatches

    h5_file = h5py.File(h5_path, 'r')
    H, W = h5_file.attrs['sensor_resolution']
    total_event_cnt = h5_file['events/xs'].shape[0]

    minipatch_size = patch_size // patch_dim
    assert patch_size % patch_dim == 0, "patch_size should be divisible by patch_dim"

    minipatch_rows = np.ceil(H / minipatch_size).astype(int)
    minipatch_cols = np.ceil(W / minipatch_size).astype(int)
    print(f"minipatch_rows: {minipatch_rows}, minipatch_cols: {minipatch_cols}")
    
    minipatch_heights = np.array([min(minipatch_size, H - r * minipatch_size) for r in range(minipatch_rows)])
    minipatch_widths = np.array([min(minipatch_size, W - c * minipatch_size) for c in range(minipatch_cols)])
    minipixels_per_patch = minipatch_heights[:, None] * minipatch_widths[None, :]

    minipatch_events = [[[] for _ in range(minipatch_cols)] for _ in range(minipatch_rows)]
    minipatch_event_counts = np.zeros((minipatch_rows, minipatch_cols), dtype=np.int32)

    all_triggered_patches = []

    for i in range(0, total_event_cnt, event_batch_size):
        # fetch a batch of events
        xs = h5_file['events/xs'][i:i+event_batch_size]
        ys = h5_file['events/ys'][i:i+event_batch_size]
        ts = h5_file['events/ts'][i:i+event_batch_size]
        ps = h5_file['events/ps'][i:i+event_batch_size]
        
        minipatch_rs = ys // minipatch_size
        minipatch_cs = xs // minipatch_size
        
        if len(minipatch_rs) == 0:
            continue
        # update patch_event_counts
        flat_indices = minipatch_rs * minipatch_cols + minipatch_cs
        new_counts = np.bincount(flat_indices, minlength=minipatch_rows * minipatch_cols)
        minipatch_event_counts += new_counts.reshape(minipatch_rows, minipatch_cols)

        for x, y, t, p in zip(xs, ys, ts, ps):
            r = y // minipatch_size
            c = x // minipatch_size
            minipatch_events[r][c].append((x, y, t, p))

        # max_patch   
        maxr, maxc = find_max_patch(minipatch_event_counts, minipatch_rows, minipatch_cols, patch_dim * 2)
        max_patchsize = np.sum(minipatch_event_counts[maxr:maxr+patch_dim,maxc:maxc+patch_dim])
        events_list = []
        for row in minipatch_events[maxr:maxr+2*patch_dim]:
            for cell in row[maxc:maxc+2*patch_dim]:
                events_list.extend(cell)  # 如果每个cell是事件列表
        max_event = np.array(events_list).reshape(-1, 4)
        max_event = max_event[np.argsort(max_event[:,2])]  # 按时间戳排序
        if max_event.shape[0] > 0 :
            xs, ys, ts, ps = max_event[:,0], max_event[:,1], max_event[:,2], max_event[:,3]
            xs = (xs - maxc * minipatch_size).astype(np.uint64)
            ys = (ys - maxr * minipatch_size).astype(np.uint64)
            voxel = event_to_voxel_events(xs, ys, ts, ps, num_bins, patch_size*2, patch_size*2)
            ey = maxr * minipatch_size
            ex = maxc * minipatch_size
            eh = patch_size*2
            ew = patch_size*2
            all_triggered_patches.append((voxel, ey, ex, eh, ew, ts[0],ts[-1]))
            minipatch_events[maxr:maxr+patch_dim*2][maxc:maxc+patch_dim*2] = [[] for _ in range(patch_dim)]
            minipatch_event_counts[maxr:maxr+patch_dim*2,maxc:maxc+patch_dim*2] = 0
        else:
            continue
        
        
        for r in range((maxr-1)%patch_dim+1, minipatch_rows+patch_dim, patch_dim):
            for c in range((maxc-1)%patch_dim+1, minipatch_cols+patch_dim, patch_dim):
                if maxr <= r < maxr + patch_dim * 2 and maxc <= c < maxc + patch_dim * 2:
                    continue
                rl = max(r-patch_dim, 0)
                cl = max(c-patch_dim, 0)
                rr = min(r, minipatch_rows)
                cr = min(c, minipatch_cols)
                pixel = (rr-rl) * (cr-cl) * minipatch_size * minipatch_size
                pixel_event = minipatch_event_counts[rl:rr,cl:cr].sum()
                if pixel_event >= pixel * trigger_rate:
                    # print(f"Trigger patch at minipatch ({rl},{cl}) to ({rr},{cr}), event count: {pixel_event}, pixel count: {pixel}")
                    events_list = []
                    for row in minipatch_events[rl:rr]:
                        for cell in row[cl:cr]:
                            events_list.extend(cell)  # 如果每个cell是事件列表
                    events = np.array(events_list).reshape(-1, 4)
                    if events.shape[0] == 0:
                        continue
                    events = events[np.argsort(events[:,2])]  # 按时间戳排序
                    xs, ys, ts, ps = events[:,0], events[:,1], events[:,2], events[:,3]
                    xs = (xs - cl * minipatch_size).astype(np.uint64)
                    ys = (ys - rl * minipatch_size).astype(np.uint64)
                    ey = rl * minipatch_size
                    ex = cl * minipatch_size
                    eh = (rr-rl) * minipatch_size
                    ew = (cr-cl) * minipatch_size
                    voxel = event_to_voxel_events(xs, ys, ts, ps, num_bins, (eh), (ew))
                    all_triggered_patches.append((voxel, ey, ex, eh, ew, ts[0],ts[-1]))
                    for i in range(rl, rr):
                        for j in range(cl, cr):
                            minipatch_events[i][j] = []  # 将每个单元格设置为空 for _ in range(rr-rl)]
                    minipatch_event_counts[rl:rr,cl:cr] = 0
                
        padded_H = minipatch_rows * minipatch_size
        padded_W = minipatch_cols * minipatch_size
    h5_file.close()
    return all_triggered_patches, padded_H, padded_W