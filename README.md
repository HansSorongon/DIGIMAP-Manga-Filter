# Manga Filter App

The Manga Filter App is a desktop application that allows users to apply artistic manga-style filters to images, inspired by the unique aesthetic of manga art. This project aims to make manga-style image editing accessible to anyone, bridging the gap between professional tools and casual users who want to create stylized artwork. This Python-based application uses customtkinter for its user interface and provides multiple sliders and switches for real-time image processing adjustments. The app supports features such as dithering, black-and-white thresholding, and custom image scaling, enabling users to easily transform their images into manga-style creations.

## Features

1. **Real-Time Filter Adjustments**:
   - Adjust parameters like `Sigma`, `k`, `Sharpen`, `Phi`, `Epsilon`, and `Scale` for creating a manga-style effect.
   - Preview changes in real-time.

2. **Dithering**:
   - Apply a dithering effect to simulate manga-style shading.

3. **Black-and-White Thresholding**:
   - Convert images to a pure black-and-white format for stylized manga aesthetics.

4. **Image Processing**:
   - Use advanced techniques such as XDoG (eXtended Difference of Gaussians) for edge detection and sharpening.

5. **Image Scaling**:
   - Adjust the scale of the processed image.

6. **User-Friendly Interface**:
   - Intuitive sliders and switches to control effects.
   - Load and save images with file dialogs.

7. **Dark Mode**:
   - Designed with a dark theme for a sleek appearance.

## Installation

### Prerequisites

1. Python 3.7 or higher
2. Required libraries:
   - `opencv-python`
   - `numpy`
   - `customtkinter`
   - `Pillow`

### Steps

1. Clone the repository or download the source code.
2. Install the required dependencies using pip:
   ```bash
   pip install opencv-python numpy customtkinter Pillow
   ```
3. Run the application:
   ```bash
   python main.py
   ``` 
   
## How to Use

1. Launch the application.
2. Click on the **Load Image** button to select an image file.
3. Adjust sliders to tweak the manga filter parameters:
   - **Sigma**: Controls the Gaussian blur.
   - **k**: Adjusts the coefficient of the second Gaussian blur.
   - **Sharpen**: Controls the strength of sharpening.
   - **Phi**: Adjusts the intensity of the soft thresholding. Increase this for more pronounced black-and-white transitions in the image.
   - **Epsilon**: Controls the luminance threshold for white regions. Increase this for a darker image with larger dark areas.
   - **Scale**: Resizes the image.
   - **Dither Dist.**: Sets the distance between pixels for dithering.
4. Toggle switches for dithering and black-and-white thresholding.
5. Preview the filtered image in real-time.
6. Save the processed image using the **Save Image** button.  
 
