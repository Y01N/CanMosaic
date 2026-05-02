import torch
import torch.nn.functional as F
import numpy as np
import cv2
import math
import random

@torch.no_grad()
def run_ml_optimizer(
    target_image_path,
    cans_rgba,
    initial_state,
    output_path,
    can_w_px,
    can_h_px,
    iterations=1000,
    batch_size=50,
    max_stack=5
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------------------------
    # LOAD TARGET
    # ---------------------------
    target = cv2.imread(target_image_path, cv2.IMREAD_UNCHANGED)
    if target is None:
        return False

    if len(target.shape) == 2:
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
    elif target.shape[2] == 4:
        target = target[:, :, :3]

    H_hr, W_hr = target.shape[:2]
    target_bgr_hr = torch.tensor(target, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0

    # ---------------------------
    # LOW-RES SCALE
    # ---------------------------
    scale_factor = 0.25
    H_lr = max(1, int(H_hr * scale_factor))
    W_lr = max(1, int(W_hr * scale_factor))
    can_w_lr = max(1, int(can_w_px * scale_factor))
    can_h_lr = max(1, int(can_h_px * scale_factor))

    target_bgr_lr = F.interpolate(target_bgr_hr, size=(H_lr, W_lr), mode='bilinear', align_corners=False)

    # ---------------------------
    # LOAD CANS
    # ---------------------------
    cans_tensor = [
        torch.tensor(c, dtype=torch.float32).permute(2, 0, 1) / 255.0
        for c in cans_rgba
    ]
    cans_hr = torch.stack(cans_tensor).to(device)
    cans_lr = F.interpolate(cans_hr, size=(can_h_lr, can_w_lr), mode='bilinear', align_corners=False)

    K = len(initial_state)

    # ---------------------------
    # INIT STATE
    # ---------------------------
    cx = torch.zeros(batch_size, K, device=device)
    cy = torch.zeros(batch_size, K, device=device)
    theta = torch.zeros(batch_size, K, device=device)
    can_id = torch.zeros(batch_size, K, dtype=torch.long, device=device)
    z_order = torch.zeros(batch_size, K, dtype=torch.long, device=device)

    initial_state.sort(key=lambda x: x['z_layer'])

    for k, state in enumerate(initial_state):
        cx[:, k] = state['cx'] * scale_factor
        cy[:, k] = state['cy'] * scale_factor
        theta[:, k] = state['rotation']
        can_id[:, k] = state['can_idx']
        z_order[:, k] = k

    print(f"Optimizing {K} cans...")

    # ===========================
    # MAIN LOOP
    # ===========================
    for i in range(iterations):

        # -----------------------
        # MUTATION (STRONGER)
        # -----------------------
        mask = (torch.rand(batch_size - 1, K, device=device) < 0.15).float()

        # 🔥 stronger micro moves
        cx[1:] += torch.randn(batch_size - 1, K, device=device) * (can_w_lr * 0.5) * mask
        cy[1:] += torch.randn(batch_size - 1, K, device=device) * (can_h_lr * 0.5) * mask

        # 🔥 stronger macro moves
        macro_mask = (torch.rand(batch_size - 1, K, device=device) < 0.05).float()
        cx[1:] += torch.randn(batch_size - 1, K, device=device) * (can_w_lr * 5.0) * macro_mask
        cy[1:] += torch.randn(batch_size - 1, K, device=device) * (can_h_lr * 5.0) * macro_mask

        cx = torch.clamp(cx, 0, W_lr - 1)
        cy = torch.clamp(cy, 0, H_lr - 1)

        # rotation
        theta[1:] += torch.randn(batch_size - 1, K, device=device) * 0.4

        # -----------------------
        # RENDER
        # -----------------------
        canvas = torch.zeros(batch_size, 4, H_lr, W_lr, device=device)
        batch_idx = torch.arange(batch_size, device=device)

        for layer in range(K):
            idx = z_order[:, layer]

            c_x = cx[batch_idx, idx]
            c_y = cy[batch_idx, idx]
            t = theta[batch_idx, idx]
            c_id = can_id[batch_idx, idx]

            can_imgs = cans_lr[c_id]

            cos_t = torch.cos(t)
            sin_t = torch.sin(t)

            # ✅ CORRECT TRANSFORM
            tx = (c_x / (W_lr - 1)) * 2 - 1
            ty = (c_y / (H_lr - 1)) * 2 - 1

            sx = can_w_lr / W_lr
            sy = can_h_lr / H_lr

            M = torch.zeros(batch_size, 2, 3, device=device)

            M[:, 0, 0] = cos_t * sx
            M[:, 0, 1] = -sin_t * sy
            M[:, 1, 0] = sin_t * sx
            M[:, 1, 1] = cos_t * sy

            M[:, 0, 2] = tx
            M[:, 1, 2] = ty

            grid = F.affine_grid(M, size=(batch_size, 4, H_lr, W_lr), align_corners=False)
            warped = F.grid_sample(can_imgs, grid, align_corners=False)

            alpha = warped[:, 3:4]
            canvas[:, :3] = warped[:, :3] * alpha + canvas[:, :3] * (1 - alpha)
            canvas[:, 3:4] = alpha + canvas[:, 3:4] * (1 - alpha)

        # -----------------------
        # LOSS
        # -----------------------
        target_expand = target_bgr_lr.expand(batch_size, 3, H_lr, W_lr)

        final = canvas[:, :3] * canvas[:, 3:4] + target_expand * (1 - canvas[:, 3:4])

        loss = F.mse_loss(final, target_expand, reduction='none').mean(dim=(1,2,3))

        best_idx = torch.argmin(loss)

        cx = cx[best_idx].unsqueeze(0).repeat(batch_size, 1)
        cy = cy[best_idx].unsqueeze(0).repeat(batch_size, 1)
        theta = theta[best_idx].unsqueeze(0).repeat(batch_size, 1)
        can_id = can_id[best_idx].unsqueeze(0).repeat(batch_size, 1)
        z_order = z_order[best_idx].unsqueeze(0).repeat(batch_size, 1)

        if (i+1) % 10 == 0:
            print(f"Iter {i+1}: Loss = {loss[best_idx]:.5f}")

    # ===========================
    # FINAL RENDER (FIXED)
    # ===========================
    print("Rendering final...")
    canvas_hr = torch.zeros(1, 4, H_hr, W_hr, device=device)

    for layer in range(K):
        idx = z_order[0, layer]

        c_x = cx[0, idx] / scale_factor
        c_y = cy[0, idx] / scale_factor
        t = theta[0, idx]
        c_id = can_id[0, idx]

        can_img = cans_hr[c_id].unsqueeze(0)

        cos_t = torch.cos(t)
        sin_t = torch.sin(t)

        tx = (c_x / (W_hr - 1)) * 2 - 1
        ty = (c_y / (H_hr - 1)) * 2 - 1

        sx = can_w_px / W_hr
        sy = can_h_px / H_hr

        M = torch.zeros(1, 2, 3, device=device)

        M[:, 0, 0] = cos_t * sx
        M[:, 0, 1] = -sin_t * sy
        M[:, 1, 0] = sin_t * sx
        M[:, 1, 1] = cos_t * sy

        M[:, 0, 2] = tx
        M[:, 1, 2] = ty

        grid = F.affine_grid(M, size=(1, 4, H_hr, W_hr), align_corners=False)
        warped = F.grid_sample(can_img, grid, align_corners=False)

        alpha = warped[:, 3:4]
        canvas_hr[:, :3] = warped[:, :3] * alpha + canvas_hr[:, :3] * (1 - alpha)
        canvas_hr[:, 3:4] = alpha + canvas_hr[:, 3:4] * (1 - alpha)

    final = (canvas_hr.squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(output_path, final)

    return True