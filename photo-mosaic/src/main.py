import os
from PIL import Image
from mosaic.mosaic_generator import MosaicGenerator

def main():
    input_image_path = input("Enter the path to the input image: ")
    output_size_inches = float(input("Enter the desired output size in inches: "))
    photo_folder = input("Enter the path to the folder containing photos: ")

    # Convert inches to pixels (assuming square output and 300 DPI)
    dpi = 300
    output_size_pixels = (int(output_size_inches * dpi), int(output_size_inches * dpi))

    mosaic_generator = MosaicGenerator()
    mosaic_image = mosaic_generator.create_mosaic(input_image_path, output_size_pixels, photo_folder)
    print("Mosaic created and saved as 'mosaic_output.png'.")

if __name__ == "__main__":
    main()