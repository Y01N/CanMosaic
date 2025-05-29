class MosaicGenerator:
    def __init__(self):
        pass

    def create_mosaic(self, input_image, output_size, photo_folder):
        from PIL import Image
        import os
        import random

        # Load the input image
        target_image = Image.open(input_image)
        target_width, target_height = target_image.size

        # Calculate the number of photos needed based on the output size
        output_width, output_height = output_size
        photo_width, photo_height = 4.83 * 300, 8.16 * 300  # Convert inches to pixels (assuming 300 DPI)
        num_photos_x = int(output_width / photo_width)
        num_photos_y = int(output_height / photo_height)

        # Load images from the specified folder
        images = self.load_images_from_folder(photo_folder)

        # Create a blank canvas for the mosaic
        mosaic_image = Image.new('RGB', (output_width, output_height))

        # Place images on the mosaic
        for y in range(num_photos_y):
            for x in range(num_photos_x):
                if images:
                    img = random.choice(images)
                    img = img.rotate(random.choice([0, 90, 180, 270]))  # Rotate image randomly
                    mosaic_image.paste(img, (int(x * photo_width), int(y * photo_height)))

        # Save the final mosaic image
        mosaic_image.save('mosaic_output.png')

    def load_images_from_folder(self, folder):
        from PIL import Image
        import os
        images = []
        for filename in os.listdir(folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder, filename)
                images.append(Image.open(img_path))
        return images