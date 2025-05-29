import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import re

DPI = 300

def in_to_px(inches):
    return int(round(inches * DPI))

def load_target_edge_map(image_path, canvas_width_in, canvas_height_in):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (in_to_px(canvas_width_in), in_to_px(canvas_height_in)))
    return cv2.Canny(img, 100, 200)

def parse_can_images_from_folders(root_dir):
    """
    Scans a directory for subfolders named like '8.16x4.83' and returns list of (image_path, (h, w) in inches).
    """
    can_images = []
    root = Path(root_dir)

    for subfolder in root.iterdir():
        if not subfolder.is_dir():
            continue
        match = re.match(r"(\d+\.?\d*)x(\d+\.?\d*)", subfolder.name)
        if not match:
            print(f"Skipping folder {subfolder.name}: does not match WxH format")
            continue
        height_in, width_in = float(match.group(1)), float(match.group(2))

        for img_path in subfolder.glob("*.png"):
            can_images.append((img_path, (height_in, width_in)))

    return can_images

def energy_function(canvas, edge_map):
    canvas_gray = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGBA2GRAY)
    canvas_edges = cv2.Canny(canvas_gray, 100, 200)

    missing_edges = np.sum((edge_map > 0) & (canvas_edges == 0))

    alpha = np.array(canvas.split()[-1])
    overlap_penalty = np.sum(alpha > 255)
    gap_penalty = np.sum((alpha == 0) & (edge_map > 0))

    total_energy = missing_edges + 5 * gap_penalty + 0.1 * overlap_penalty
    return total_energy
