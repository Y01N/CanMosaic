import os
from PIL import Image
from pathlib import Path
from .utils import (
    load_target_edge_map,
    overlay_edges_on_image,
    parse_can_images_from_folders,
    in_to_px,
    energy_function
)

class MosaicCreator:
    def __init__(self, target_image_path, can_image_root, canvas_width_in, canvas_height_in, debug=False):
        self.debug = debug
        self.target_image_path = target_image_path
        self.canvas_width_in = canvas_width_in
        self.canvas_height_in = canvas_height_in
        self.can_image_root = can_image_root
        self.DPI = 300

        from .utils import (
            load_target_edge_map,
            parse_can_images_from_folders,
            in_to_px
        )

        self.target_edges = load_target_edge_map(target_image_path, canvas_width_in, canvas_height_in)
        self.can_images = parse_can_images_from_folders(can_image_root)
        self.canvas = Image.new("RGBA", (in_to_px(canvas_width_in), in_to_px(canvas_height_in)), (255, 255, 255, 0))
        
        if self.debug:
            self.save_debug_images("debug-images")

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

    def save_debug_images(self, output_dir="debug-images"):
        os.makedirs(output_dir, exist_ok=True)

        # Resize target image to canvas size before overlay
        target_img = Image.open(self.target_image_path).convert("RGB")
        canvas_px_size = (in_to_px(self.canvas_width_in), in_to_px(self.canvas_height_in))
        target_img = target_img.resize(canvas_px_size)

        overlaid_target = overlay_edges_on_image(target_img, self.target_edges)
        overlaid_target.save(os.path.join(output_dir, "target_edges_overlay.png"))

        # Save edge map overlaid on canvas
        canvas_edges = self._compute_canvas_edges()
        overlaid_canvas = overlay_edges_on_image(self.canvas.convert("RGB"), canvas_edges)
        overlaid_canvas.save(os.path.join(output_dir, "canvas_edges_overlay.png"))


    def _compute_canvas_edges(self):
        from .utils import get_edge_map_from_pil
        return get_edge_map_from_pil(self.canvas)
