from mosaic.mosaic_creator import MosaicCreator

if __name__ == "__main__":
    creator = MosaicCreator(
    target_image_path="test-images/penguin.png",
    can_image_root="test-cans/",
    canvas_width_in=24,
    canvas_height_in=36,
    debug=True  # Set to False to skip saving debug images
)
    
    creator.place_sample_can((5, 5), 15)
    energy = creator.evaluate_layout()
    print(f"Energy: {energy}")
    creator.save("output/output_layout.png")
    creator.save_debug_images("debug-images")

