# Photo Mosaic Project

This project creates a mosaic image from a collection of photos. The user can specify an input image, the desired output size in inches, and a folder containing the photos to be used in the mosaic.

The images used need to be given scale in inches via user input or by folder / file names
The target image and the input images are processed, turned into shapes and colors.

The input images are then arranged to make up the target image. The input images are cans irl.
The arranged cans may be placed anywhere on the canvas.
Parameters / creative decisions
Can the cans be rotated in any manner - Yes
Can the cans overlap eachother and are there limits on how many can stack - Yes they can overlap and yes there is a limit bout 5-10 max stack
Can the cans folded or cut (maybe once or just at edges) - ❌
Is the edge of the area strict or can cans dangle off the border of the given canvas?  - can dangle
Must the cans cover the entire canvas - It should try to
Can you repeat cans - only for testing

The algorithm that places cans should optimize for 
- matching edges/shapes
- making groups of colors from source image a uniform color
- matching colors or shades (grayscale colors)
- covering the canvas
- minimizing can overlap 
- maybe maximizing size of visible can

## Project Structure

```
photo-mosaic
├── src
│   ├── main.py
│   ├── mosaic
│   │   ├── __init__.py
│   │   ├── mosaic_generator.py
│   │   └── image_utils.py
├── requirements.txt
└── README.md
```

## Installation

To set up the project, you need to install the required dependencies. You can do this by running:

```
pip install -r requirements.txt
```

## Usage

1. Prepare your input image and a folder containing the photos you want to use for the mosaic.
2. Run the main program:

```
python src/main.py
```

3. Follow the prompts to input the path to your image, the desired output size in inches, and the folder containing the photos.

## Functionality

- **Input Image**: The image that will be transformed into a mosaic.
- **Output Size**: The size of the output mosaic image in inches.
- **Photo Folder**: A directory containing images that will be used to create the mosaic.

## Modules

- `mosaic_generator.py`: Contains the `MosaicGenerator` class responsible for generating the mosaic.
- `image_utils.py`: Provides utility functions for loading images and rotating them as needed.

## License

This project is licensed under the MIT License.