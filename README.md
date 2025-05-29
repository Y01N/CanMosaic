# Photo Mosaic Project

This project creates a mosaic image from a collection of photos. The user can specify an input image, the desired output size in inches, and a folder containing the photos to be used in the mosaic.

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