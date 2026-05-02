import os
import glob
import cv2
import numpy as np
import torch
import random
import math

def load_cans(cans_folder, can_w_px, can_h_px):
    can_paths = glob.glob(os.path.join(cans_folder, '*'))
    cans_rgba = []
    cans_bgr_mean = []
    
    for path in can_paths:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None: continue
            
            img = cv2.resize(img, (can_w_px, can_h_px))
            
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                
            alpha = img[:, :, 3]
            mask = alpha > 0
            if not np.any(mask): continue
                
            bgr = img[:, :, :3]
            mean_color = np.mean(bgr[mask], axis=0)
            
            cans_rgba.append(img)
            cans_bgr_mean.append(mean_color)
        except Exception:
            pass
            
    return cans_rgba, np.array(cans_bgr_mean)

def poisson_disk_sampling(width, height, radius, k=30):
    cell_size = radius / math.sqrt(2)
    grid_width = int(math.ceil(width / cell_size))
    grid_height = int(math.ceil(height / cell_size))
    
    grid = [[-1 for _ in range(grid_width)] for _ in range(grid_height)]
    
    points = []
    active_list = []
    
    x0 = random.uniform(0, width)
    y0 = random.uniform(0, height)
    
    points.append((x0, y0))
    active_list.append(0)
    grid[min(int(y0 / cell_size), grid_height - 1)][min(int(x0 / cell_size), grid_width - 1)] = 0
    
    while active_list:
        idx = random.choice(range(len(active_list)))
        point_idx = active_list[idx]
        px, py = points[point_idx]
        
        found = False
        for _ in range(k):
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(radius, 2 * radius)
            
            nx = px + r * math.cos(angle)
            ny = py + r * math.sin(angle)
            
            if 0 <= nx < width and 0 <= ny < height:
                grid_x = min(int(nx / cell_size), grid_width - 1)
                grid_y = min(int(ny / cell_size), grid_height - 1)
                
                too_close = False
                x_min = max(0, grid_x - 2)
                x_max = min(grid_width - 1, grid_x + 2)
                y_min = max(0, grid_y - 2)
                y_max = min(grid_height - 1, grid_y + 2)
                
                for gy in range(y_min, y_max + 1):
                    for gx in range(x_min, x_max + 1):
                        neighbor_idx = grid[gy][gx]
                        if neighbor_idx != -1:
                            nx_o, ny_o = points[neighbor_idx]
                            dist_sq = (nx - nx_o)**2 + (ny - ny_o)**2
                            if dist_sq < radius**2:
                                too_close = True
                                break
                    if too_close:
                        break
                        
                if not too_close:
                    points.append((nx, ny))
                    new_idx = len(points) - 1
                    active_list.append(new_idx)
                    grid[grid_y][grid_x] = new_idx
                    found = True
                    break
                    
        if not found:
            active_list.pop(idx)
            
    return points

def run_greedy_template(target_image_path, cans_folder, output_path=None, can_w_px=50, can_h_px=50, emoji_multiplier=1.0):
    """
    Acts as the initialization engine for the ML Optimizer.
    Uses Poisson Disk Sampling to generate high-coverage coordinates, and Bipartite Matching for colors.
    """
    cans_rgba, cans_bgr_mean = load_cans(cans_folder, can_w_px, can_h_px)
    if not cans_rgba:
        return []
        
    target = cv2.imread(target_image_path, cv2.IMREAD_UNCHANGED)
    if target is None:
        return []
        
    if len(target.shape) == 2:
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
    elif target.shape[2] == 4:
        target = target[:, :, :3]
        
    H, W, _ = target.shape
    
    # 1. Generate High-Coverage Coordinates
    radius = (min(can_w_px, can_h_px) * 0.75) / math.sqrt(emoji_multiplier)
    points = poisson_disk_sampling(W, H, radius)
    
    print(f"  Initialized {len(points)} placement coordinates.")
    
    # 2. Extract Average Colors
    patches_mean = []
    valid_points = []
    
    for px, py in points:
        x_start = int(px - can_w_px / 2)
        y_start = int(py - can_h_px / 2)
        
        cx_start = max(0, x_start)
        cx_end = min(W, x_start + can_w_px)
        cy_start = max(0, y_start)
        cy_end = min(H, y_start + can_h_px)
        
        if cx_end <= cx_start or cy_end <= cy_start: continue
        
        patch = target[cy_start:cy_end, cx_start:cx_end]
        if patch.size > 0:
            patches_mean.append(np.mean(patch, axis=(0,1)))
            valid_points.append((px, py))
            
    # 3. Unique Color Assignment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = torch.tensor(np.array(patches_mean), dtype=torch.float32, device=device)
    C = torch.tensor(cans_bgr_mean, dtype=torch.float32, device=device)
    
    distances = torch.cdist(G, C)
    num_points, num_cans = distances.shape
    
    flat_indices = torch.argsort(distances.view(-1)).cpu().numpy()
    
    used_points = set()
    used_cans = set()
    best_can_indices = np.full(num_points, -1, dtype=int)
    
    for idx in flat_indices:
        pt_idx = int(idx // num_cans)
        can_idx = int(idx % num_cans)
        
        if pt_idx not in used_points and can_idx not in used_cans:
            best_can_indices[pt_idx] = can_idx
            used_points.add(pt_idx)
            used_cans.add(can_idx)
            if len(used_points) == num_points or len(used_cans) == num_cans: break
            
    # 4. Generate Initial State and Preview Image
    initial_state = []
    canvas = np.zeros((H, W, 4), dtype=np.uint8)
    z_idx = 0
    
    for i, (cx, cy) in enumerate(valid_points):
        can_idx = best_can_indices[i]
        if can_idx == -1: continue
        
        initial_state.append({
            "can_idx": int(can_idx),
            "cx": float(cx),
            "cy": float(cy),
            "rotation": 0.0,
            "z_layer": int(z_idx)
        })
        z_idx += 1
        
        # Render onto canvas for preview
        x_start = int(cx - can_w_px / 2)
        y_start = int(cy - can_h_px / 2)
        x_end = x_start + can_w_px
        y_end = y_start + can_h_px
        
        cx_start = max(0, x_start)
        cx_end = min(W, x_end)
        cy_start = max(0, y_start)
        cy_end = min(H, y_end)
        
        can_img = cans_rgba[can_idx]
        can_sx_start = cx_start - x_start
        can_sx_end = can_sx_start + (cx_end - cx_start)
        can_sy_start = cy_start - y_start
        can_sy_end = can_sy_start + (cy_end - cy_start)
        
        can_slice = can_img[can_sy_start:can_sy_end, can_sx_start:can_sx_end, :]
        can_alpha = can_slice[:, :, 3] / 255.0
        canvas_alpha = canvas[cy_start:cy_end, cx_start:cx_end, 3] / 255.0
        
        out_alpha = can_alpha + canvas_alpha * (1 - can_alpha)
        out_alpha_safe = np.where(out_alpha == 0, 1.0, out_alpha)
        
        for c in range(3):
            canvas[cy_start:cy_end, cx_start:cx_end, c] = (
                can_slice[:, :, c] * can_alpha + 
                canvas[cy_start:cy_end, cx_start:cx_end, c] * canvas_alpha * (1 - can_alpha)
            ) / out_alpha_safe
            
        canvas[cy_start:cy_end, cx_start:cx_end, 3] = out_alpha * 255

    if output_path is not None:
        cv2.imwrite(output_path, canvas)
        
    return initial_state

if __name__ == "__main__":
    pass
