def load_images_from_folder(folder):
    import os
    from PIL import Image

    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            images.append(Image.open(img_path))
    return images

def rotate_image(image, angle):
    return image.rotate(angle, expand=True)