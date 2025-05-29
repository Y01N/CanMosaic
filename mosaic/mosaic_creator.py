from PIL import Image
from pathlib import Path
from .utils import (
    load_target_edge_map,
    parse_can_images_from_folders,
    in_to_px,
    energy_function
)

class MosaicCreator:
    def __init__(self, target_image_path, can_image_root, canvas_width_in, canvas_height_in):
        self.DPI = 300
        self.canvas_width_in = canvas_width_in
        self.canvas_height_in = canvas_height_in
        self.canvas = Image.new("RGBA", (in_to_px(canvas_width_in), in_to_px(canvas_height_in)), (255, 255, 255, 0))

        self.target_edges = load_target_edge_map(target_image_path, canvas_width_in, canvas_height_in)
        self.can_images = parse_can_images_from_folders(can_image_root)

    def place_sample_can(self, position_in, rotation_deg):
        can_path, size = self.can_images[0]
        can_img = Image.open(can_path).convert("RGBA")

        x_px = in_to_px(position_in[0])
        y_px = in_to_px(position_in[1])
        rotated = can_img.rotate(rotation_deg, expand=True)
        self.canvas.paste(rotated, (x_px, y_px), rotated)

    def evaluate_layout(self):
        return energy_function(self.canvas, self.target_edges)

    def save(self, output_path):
        self.canvas.save(output_path)
