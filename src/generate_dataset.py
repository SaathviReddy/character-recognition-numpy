"""
Character Dataset Generator

This program creates a synthetic image dataset containing
35 character classes:

    - Uppercase letters A-Z
    - Digits 1-9

Each class contains 100 grayscale images of size 28x28 pixels.

Small variations in font, size, position, and rotation are
introduced to make the dataset more diverse.
"""

import os
import random

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Configuration
# ============================================================

# 26 uppercase letters + 9 digits = 35 classes
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("123456789")

# Number of images generated for every class
SAMPLES_PER_CLASS = 100

# All images will have dimensions 28 x 28 pixels
IMAGE_SIZE = 28

# Location where the generated images will be stored
OUTPUT_DIR = os.path.join("data", "raw")

# Windows font directory
FONT_DIR = r"C:\Windows\Fonts"

# Fonts that we checked are available on the computer
FONT_FILES = [
    "arial.ttf",
    "calibri.ttf",
    "times.ttf",
    "cour.ttf",
    "georgia.ttf",
]


# ============================================================
# Font Functions
# ============================================================

def get_available_fonts():
    """
    Find all usable fonts from the configured font directory.

    Returns
    -------
    list
        A list containing the full paths of available font files.
    """

    available_fonts = []

    for font_file in FONT_FILES:
        font_path = os.path.join(FONT_DIR, font_file)

        if os.path.exists(font_path):
            available_fonts.append(font_path)

    return available_fonts


# ============================================================
# Image Generation Function
# ============================================================

def generate_character_image(character, font_path):
    """
    Generate one grayscale image containing a character.

    Parameters
    ----------
    character : str
        Character to draw, such as 'A' or '5'.

    font_path : str
        Path to the font used to draw the character.

    Returns
    -------
    PIL.Image.Image
        A generated 28x28 grayscale image.
    """

    # Create a black 28x28 grayscale image.
    image = Image.new(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        0
    )

    # Create a drawing object.
    draw = ImageDraw.Draw(image)

    # Randomly choose the character size.
    font_size = random.randint(18, 24)

    # Load the selected font.
    font = ImageFont.truetype(
        font_path,
        font_size
    )

    # Find the dimensions of the character.
    bounding_box = draw.textbbox(
        (0, 0),
        character,
        font=font
    )

    text_width = bounding_box[2] - bounding_box[0]
    text_height = bounding_box[3] - bounding_box[1]

    # Calculate the position needed to approximately
    # center the character.
    x = (
        IMAGE_SIZE - text_width
    ) // 2 - bounding_box[0]

    y = (
        IMAGE_SIZE - text_height
    ) // 2 - bounding_box[1]

    # Add a small random shift.
    x += random.randint(-2, 2)
    y += random.randint(-2, 2)

    # Draw the character in white.
    draw.text(
        (x, y),
        character,
        fill=255,
        font=font
    )

    # Randomly rotate the image slightly.
    rotation_angle = random.uniform(-10, 10)

    image = image.rotate(
        rotation_angle,
        resample=Image.Resampling.BILINEAR,
        fillcolor=0
    )

    return image


# ============================================================
# Dataset Generation
# ============================================================

def generate_dataset():
    """
    Generate the complete 35-class character dataset.

    For every class, 100 images are generated and stored
    in a separate directory under data/raw/.
    """

    # Find available fonts.
    available_fonts = get_available_fonts()

    if not available_fonts:
        raise RuntimeError(
            "No usable fonts were found."
        )

    print(f"Found {len(available_fonts)} usable fonts.")
    print(f"Number of classes: {len(CLASSES)}")
    print(f"Samples per class: {SAMPLES_PER_CLASS}")
    print(
        f"Total images to generate: "
        f"{len(CLASSES) * SAMPLES_PER_CLASS}"
    )

    # Process every character class.
    for character in CLASSES:

        # Create a directory for the current class.
        class_directory = os.path.join(
            OUTPUT_DIR,
            character
        )

        os.makedirs(
            class_directory,
            exist_ok=True
        )

        print(
            f"Generating class: {character}"
        )

        # Generate the required number of images.
        for index in range(SAMPLES_PER_CLASS):

            # Randomly choose one of the available fonts.
            font_path = random.choice(
                available_fonts
            )

            # Generate one image.
            image = generate_character_image(
                character,
                font_path
            )

            # Create a filename such as A_000.png.
            filename = (
                f"{character}_{index:03d}.png"
            )

            # Create the complete output path.
            output_path = os.path.join(
                class_directory,
                filename
            )

            # Save the image.
            image.save(output_path)


# ============================================================
# Main Program
# ============================================================

if __name__ == "__main__":
    generate_dataset()