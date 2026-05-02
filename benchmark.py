import time
import cv2
import numpy as np
import os
import glob
from skimage.metrics import structural_similarity as ssim

# Import the new architecture
from greedy_template import run_greedy_template, load_cans
from ml_optimizer import run_ml_optimizer

def compute_metrics(target_path, result_path):
    target = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)
    result = cv2.imread(result_path, cv2.IMREAD_UNCHANGED)
    
    if result is None:
        return {"error": "Result image not produced."}

    if target.shape[:2] != result.shape[:2]:
        result = cv2.resize(result, (target.shape[1], target.shape[0]))

    if result.shape[2] == 4:
        alpha = result[:, :, 3]
        coverage = np.sum(alpha > 0) / (result.shape[0] * result.shape[1])
    else:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        coverage = np.sum(gray > 0) / (result.shape[0] * result.shape[1])

    target_rgb = cv2.cvtColor(target, cv2.COLOR_BGRA2RGB) if target.shape[2] == 4 else cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGRA2RGB) if result.shape[2] == 4 else cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    mse = np.mean((target_rgb.astype(np.float32) - result_rgb.astype(np.float32)) ** 2)

    min_dim = min(target_rgb.shape[0], target_rgb.shape[1])
    win_size = min(7, min_dim)
    if win_size % 2 == 0:
        win_size -= 1
        
    ssim_val = ssim(target_rgb, result_rgb, channel_axis=2, data_range=255, win_size=max(3, win_size))

    return {
        "coverage_pct": coverage * 100.0,
        "mse": mse,
        "ssim": ssim_val
    }

def run_benchmarks():
    target_images = glob.glob("images/target-images/*")
    if not target_images:
        print("Error: No target images found")
        return
    
    target_images = target_images[:3]
    cans_folder = "images/emojis"
    
    # === CONFIGURATION PARAMS ===
    canvas_width_in = 8.5
    canvas_height_in = 10.0
    can_width_in = 0.5
    can_height_in = 0.5
    base_dpi = 100
    
    # EMOJI_MULTIPLIER controls density. 1.0 = normal coverage, 2.0 = twice as many emojis.
    EMOJI_MULTIPLIER = 2.0 
    
    # Extent of ML Optimization (Higher = better quality, but takes longer)
    ML_ITERATIONS = 500
    ML_BATCH_SIZE = 50
    
    # ML Optimization takes time, keep scales reasonable
    scales = [0.25, 0.5]

    for target_image in target_images:
        target_name = os.path.splitext(os.path.basename(target_image))[0]
        print(f"\n{'#'*60}")
        print(f"BENCHMARKING TARGET: {target_name}")
        print(f"{'#'*60}")
        
        orig_target = cv2.imread(target_image, cv2.IMREAD_UNCHANGED)
        if orig_target is None: continue

        for scale in scales:
            current_dpi = int(base_dpi * scale)
            
            canvas_w_px = int(canvas_width_in * current_dpi)
            canvas_h_px = int(canvas_height_in * current_dpi)
            can_w_px = max(1, int(can_width_in * current_dpi))
            can_h_px = max(1, int(can_height_in * current_dpi))

            out_dir = os.path.join("benchmark_results_ML", target_name, f"{scale}x")
            os.makedirs(out_dir, exist_ok=True)

            print(f"\n{'='*50}")
            print(f"RUNNING AT {scale}x SCALE (DPI: {current_dpi})")
            print(f"{'='*50}")
            
            scaled_target_path = os.path.join(out_dir, "target.png")
            cv2.imwrite(scaled_target_path, cv2.resize(orig_target, (canvas_w_px, canvas_h_px)))

            # Step 1: Initialization
            print("\nStep 1: Greedy Initialization...")
            start_init = time.time()
            greedy_out_path = os.path.join(out_dir, "Greedy_Init.png")
            
            initial_state = run_greedy_template(scaled_target_path, cans_folder, output_path=greedy_out_path, 
                                                can_w_px=can_w_px, can_h_px=can_h_px, emoji_multiplier=EMOJI_MULTIPLIER)
            end_init = time.time()
            print(f"  Init Time: {end_init - start_init:.2f} seconds")
            
            if not initial_state:
                continue
                
            metrics_greedy = compute_metrics(scaled_target_path, greedy_out_path)
            print(f"  Greedy Coverage: {metrics_greedy['coverage_pct']:.1f}% | SSIM: {metrics_greedy['ssim']:.4f}")

            # Step 2: ML Optimization
            print("\nStep 2: ML Optimization (Stochastic Hill Climbing)...")
            start_ml = time.time()
            ml_out_path = os.path.join(out_dir, "ML_Optimized.png")
            
            cans_rgba, _ = load_cans(cans_folder, can_w_px, can_h_px)
            run_ml_optimizer(scaled_target_path, cans_rgba, initial_state, ml_out_path, can_w_px, can_h_px, 
                             iterations=ML_ITERATIONS, batch_size=ML_BATCH_SIZE, max_stack=5)
            end_ml = time.time()
            print(f"  ML Optim Time: {end_ml - start_ml:.2f} seconds")
            
            metrics_ml = compute_metrics(scaled_target_path, ml_out_path)
            print(f"  ML Final Coverage: {metrics_ml['coverage_pct']:.1f}% | Final SSIM: {metrics_ml['ssim']:.4f}")

    print("\nML Pipeline Benchmarking Complete!")

if __name__ == "__main__":
    run_benchmarks()
