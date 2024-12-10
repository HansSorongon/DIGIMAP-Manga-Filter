import os
import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox, PhotoImage
from PIL import Image, ImageTk

"""
MangaFilterApp: A GUI application for applying artistic filters to images, 
with features like edge enhancement, dithering, black-and-white thresholding, 
and parameterized adjustments using sliders.

Attributes:
    root (tk.Tk): The main application window.
    main_frame (CTkFrame): The container for the main content.
    left_frame (CTkFrame): Frame containing the image display area.
    right_frame (CTkFrame): Frame containing control sliders, switches, and buttons.
    is_threshold (bool): Indicates whether the black-and-white thresholding is enabled.
    image_label (CTkLabel): Label for displaying the processed image.
    sliders (dict): Stores slider components and their labels for real-time adjustment.
    load_button (CTkButton): Button to load an image from the file system.
    save_button (CTkButton): Button to save the processed image to the file system.
    original_image (numpy.ndarray): Stores the original grayscale image.
    processed_image (numpy.ndarray): Stores the processed image after applying filters.
    is_dithering (bool): Indicates whether dithering is enabled.
"""
class MangaFilterApp:

    """
    Initializes the MangaFilterApp with a GUI layout, sliders, and controls 
    for image filtering and processing.

    Args:
        root (tk.Tk): The main tkinter window object to attach the application's UI.

    Initializes:
        - A main frame containing the left and right sections of the UI.
        - A left frame for displaying images.
        - A right frame with sliders for filter parameters and buttons for image actions.
        - Sliders to adjust parameters like sigma, sharpening, and dithering distance.
        - Toggles for dithering and black-and-white thresholding.
        - Buttons to load and save images.
        - Variables to store the original and processed images, and flags for dithering and thresholding.
    """
    def __init__(self, root):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.root = root
        self.root.title("Manga Filter App")
        self.root.geometry("1200x600")

        self.main_frame = ctk.CTkFrame(root)
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        self.left_frame = ctk.CTkFrame(self.main_frame)
        self.left_frame.pack(side="left", padx=10, pady=10, fill="both", expand=True)

        self.is_threshold = False

        self.image_label = ctk.CTkLabel(
            self.left_frame, 
            text="", 
            width=400, 
            height=400
        )
        self.image_label.pack(padx=10, pady=10)

        self.right_frame = ctk.CTkFrame(self.main_frame)
        self.right_frame.pack(side="right", padx=10, pady=10, fill="y")

        self.sliders = {}
        slider_configs = [
            ("Sigma", 0, 15, 3, 10),
            ("k", 0, 100, 20, 10),
            ("Sharpen", 0, 50, 20, 1),
            ("Phi", 0, 50, 10, 1),
            ("Epsilon", 0, 50, 15, 100),
            ("Scale", 1, 100, 100, 1),
            ("Dither Dist.", 1, 10, 5, 1)
        ]

        for name, min_val, max_val, default, divisor in slider_configs:
            self.create_slider(name, min_val, max_val, default, divisor)

        self.create_dither_switch()
        self.create_threshold_switch()

        self.load_button = ctk.CTkButton(
            self.right_frame, 
            text="Load Image", 
            command=self.load_image
        )
        self.load_button.pack(padx=10, pady=5)

        self.save_button = ctk.CTkButton(
            self.right_frame, 
            text="Save Image", 
            command=self.save_image,
            state="disabled"
        )
        self.save_button.pack(padx=10, pady=5)

        self.original_image = None
        self.processed_image = None
        self.is_dithering = False

    """
    Creates a toggle switch for enabling or disabling dithering.

    This method sets up a labeled switch in the right frame of the GUI, 
    allowing the user to toggle the dithering effect on the processed image.

    The switch updates its state dynamically and triggers the `toggle_dither` 
    method when toggled.
    """
    def create_dither_switch(self):
        switch_frame = ctk.CTkFrame(self.right_frame)
        switch_frame.pack(padx=10, pady=5, fill="x")
        
        label = ctk.CTkLabel(switch_frame, text="Dithering")
        label.pack(side="left", padx=(0, 10))
        
        self.dither_switch = ctk.CTkSwitch(
            switch_frame, 
            text="Off",
            command=self.toggle_dither
        )
        self.dither_switch.pack(side="right")


    """
    Creates a toggle switch for enabling or disabling black-and-white thresholding.

    This method sets up a labeled switch in the right frame of the GUI, 
    allowing the user to apply or remove a black-and-white threshold effect 
    on the processed image.

    The switch updates its state dynamically and triggers the `toggle_threshold` 
    method when toggled.
    """
    def create_threshold_switch(self):
        switch_frame = ctk.CTkFrame(self.right_frame)
        switch_frame.pack(padx=10, pady=5, fill="x")
        
        label = ctk.CTkLabel(switch_frame, text="Black & White Threshold")
        label.pack(side="left", padx=(0, 10))
        
        self.threshold_switch = ctk.CTkSwitch(
            switch_frame, 
            text="Off",
            command=self.toggle_threshold
        )
        self.threshold_switch.pack(side="right")

    """
    Toggles the black-and-white thresholding effect for the image.

    This method switches the `is_threshold` flag between True and False, updates 
    the text of the threshold switch to indicate the current state ("On" or "Off"), 
    and reprocesses the image to apply or remove the threshold effect if an image 
    is loaded.
    """
    def toggle_threshold(self):
        self.is_threshold = not self.is_threshold
        
        self.threshold_switch.configure(
            text="On" if self.is_threshold else "Off"
        )
        
        if self.original_image is not None:
            self.process_image()

    """
    Creates a slider for adjusting a parameter in the image processing pipeline.

    Args:
        name (str): The name of the parameter being controlled by the slider.
        min_val (int): The minimum value of the slider.
        max_val (int): The maximum value of the slider.
        default (int): The default value of the slider when initialized.
        divisor (int): A divisor to scale the slider's value for finer adjustments.

    This method initializes a slider with a label and a real-time value display. 
    The slider updates the corresponding parameter in the `sliders` dictionary 
    and triggers the `update_image` method when adjusted.
    """
    def create_slider(self, name, min_val, max_val, default, divisor):
        slider_frame = ctk.CTkFrame(self.right_frame)
        slider_frame.pack(padx=10, pady=5, fill="x")

        label = ctk.CTkLabel(slider_frame, text=name)
        label.pack(side="left", padx=(0, 10))

        slider = ctk.CTkSlider(
            slider_frame, 
            from_=min_val, 
            to=max_val, 
            number_of_steps=max_val,
            command=lambda value, n=name, d=divisor: self.update_image(n, value, d)
        )
        slider.set(default)
        slider.pack(side="left", expand=True, fill="x", padx=(0, 10))

        value_label = ctk.CTkLabel(slider_frame, text=str(default))
        value_label.pack(side="right")

        self.sliders[name] = {
            'slider': slider, 
            'value_label': value_label
        }

    """
    Loads an image in grayscale, scales it, and processes it.

    Allows the user to select an image file via a dialog. Enables the "Save Image" button 
    if successful, or displays an error message on failure.

    Supported formats: JPG, JPEG, PNG, BMP, TIFF, WEBP.
    """
    def load_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")
            ]
        )
        
        if not filepath:
            return

        try:
            self.original_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
            self.original_image = self.scale_image(self.original_image, 50)

            self.process_image()

            self.save_button.configure(state="normal")

            height, width = self.original_image.shape

        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")

    """
    Saves the processed image to the file system.

    Prompts the user to specify a file location and format via a save dialog. Saves the 
    image in the selected format if a processed image exists. Displays a success 
    message on successful save or an error message on failure. Warns if no image is available.

    Supported formats: PNG, JPEG, and others.
    """
    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No image to save")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                cv2.imwrite(filepath, self.processed_image)
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save image: {str(e)}")

    """
    Updates the value of a slider and reprocesses the image.

    Args:
        name (str): The name of the slider parameter being updated.
        value (float): The current value of the slider.
        divisor (int): A scaling factor for finer adjustments.

    Updates the slider's displayed value and triggers image reprocessing if an image 
    is loaded.
    """
    def update_image(self, name, value, divisor):
        self.sliders[name]['value_label'].configure(text=str(int(value)))
        
        if self.original_image is not None:
            self.process_image()

    """
    Toggles the dithering effect on or off.

    Flips the `is_dithering` flag, updates the dither switch label to reflect 
    the current state ("On" or "Off"), and reprocesses the image if one is loaded.
    """
    def toggle_dither(self):
        self.is_dithering = not self.is_dithering
        
        self.dither_switch.configure(
            text="On" if self.is_dithering else "Off"
        )
        
        if self.original_image is not None:
            self.process_image()


    """
    Processes the loaded image using the selected filters and adjustments.

    Applies the following steps:
    1. Extracts filter parameters from sliders (e.g., sigma, sharpening, scale).
    2. Applies the XDoG (eXtended Difference of Gaussians) filter.
    3. Optionally applies black-and-white thresholding if enabled.
    4. Optionally applies dithering if enabled.
    5. Scales the processed image based on the selected scale factor.
    6. Displays the final processed image in the UI.

    Updates the `processed_image` attribute with the final result.
    """
    def process_image(self):

        sigma = self.sliders['Sigma']['slider'].get() / 10.0
        k = self.sliders['k']['slider'].get() / 10.0
        sharpen = self.sliders['Sharpen']['slider'].get()
        epsilon = self.sliders['Epsilon']['slider'].get() / 100.0
        phi = self.sliders['Phi']['slider'].get()
        scale = self.sliders['Scale']['slider'].get()
        pixel_distance = self.sliders['Dither Dist.']['slider'].get()
        threshold_value = 128  # Set the threshold value here

        # Apply the XDoG filter for artistic edge detection and enhancement.
        # The XDoG filter is an extended version of the DoG (Difference of Gaussians)
        # filter that allows for finer control over the end result allowing for better
        # artistic expression. Check the method for more details.
        result = self.xdog(
            self.original_image, 
            sigma, k, sharpen, epsilon, phi
        )

        # Binary thresholding for a pure black-and-white aesthetic (no grays). Some manga
        # artists only use black and white so making this an option helps express that style.
        if self.is_threshold:
            result = self.apply_threshold(result, threshold_value)

        # A lot of shading done in manga is done through dithering, we can "fake" that effect
        # by simply dithering lower frequency pixels under a threshold with white pixels.
        if self.is_dithering:
            result = self.dither(result, pixel_distance=int(pixel_distance))
        
        result = self.scale_image(result, scale)
        self.processed_image = result

        self.display_image(result)

    """
    Applies a binary threshold to the image.

    Args:
        image (numpy.ndarray): The input grayscale image to process.
        threshold_value (int): The threshold value for binary conversion. 
            Pixel values above this are set to 255 (white), and below are set to 0 (black).

    Returns:
        numpy.ndarray: The binary thresholded image.

    This method creates a high-contrast black-and-white image, ideal for imitating 
    certain art styles that only use pure black-and-white.
    """
    def apply_threshold(self, image, threshold_value):
        _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        return binary_image

    """
    Displays a processed image in the application UI.

    Args:
        cv_image (numpy.ndarray): The processed image in OpenCV format to be displayed.

    Converts the given image from a NumPy array to a PIL image and then to a format 
    compatible with the CTkLabel widget. Updates the `image_label` to show the processed image.
    """
    def display_image(self, cv_image):
        pil_image = Image.fromarray(cv_image)
        photo = ctk.CTkImage(light_image=pil_image, size=(pil_image.width, pil_image.height))
        
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    """
    Applies the XDoG (eXtended Difference of Gaussians) filter to create a manga-style effect.

    Args:
        image (numpy.ndarray): The input grayscale image.
        sigma_one (float): The standard deviation for the first Gaussian blur.
        k (float): The scaling factor for the second Gaussian blur.
        sharpen (float): The sharpening factor to enhance edge contrast.
        epsilon (float): The edge threshold for masking.
        phi (float): The parameter for soft thresholding to enhance line visibility.

    Returns:
        numpy.ndarray: The processed image with the XDoG filter applied.

    This method is the heart and soul of the manga filter, providing the distinctive 
    line art effect by combining Gaussian blurs, edge detection, and thresholding. 
    It highlights contours and textures, emulating the aesthetic of hand-drawn manga.
    """
    def xdog(self, image, sigma_one, k, sharpen, epsilon, phi):

        # We normalize the image to the range [0, 1] to work better with math operations.
        rescaled = image.astype(np.float32) / 255.0

        # This section computes the Difference of Gaussians filter. By subtracting two
        # blurred versions of the same image (one with a larger blur than the other), we
        # effectively get nice stylized edge detection. This works as after the subtraction,
        # regions with rapid intensity changes remain prominent, while uniform areas smooth out.
        # By doing this, we can imitate the strokes done when a manga artist draws panels. 
        img_a = cv2.GaussianBlur(rescaled, (0, 0), sigmaX=sigma_one)
        img_b = cv2.GaussianBlur(rescaled, (0, 0), sigmaX=sigma_one * k)
        scaled_diff = (sharpen + 1) * img_a - sharpen * img_b

        # We lose a lot of information doing the DoG filter so we compromise by multiplying
        # the original scaled image wih the image passed under the DoG filter to retain more of
        # the original structure of the image.
        sharpened = rescaled * scaled_diff * 255

        # Create a binary mask for significant edges based on the epsilon threshold.
        # Higher epsilon values result in fewer white regions, creating darker images with larger dark areas
        # mimicking the strong contrast seen in manga art.
        mask = (rescaled * scaled_diff > epsilon).astype(np.float32)

        # We run the result through a soft-threshold to keep more of the original shading information. This
        # information is useful for "faking" shading with dithering if needed.
        inverse_mask = 1.0 - mask
        soft_thresholded = 1.0 + np.tanh(phi * (sharpened / 255.0 - epsilon))
        result = (mask + inverse_mask * soft_thresholded) * 255.0

        return np.clip(result, 0, 255).astype(np.uint8)

    """
    Scales the image by a given percentage.

    Args:
        image (numpy.ndarray): The input image to be scaled.
        scale_percent (int): The percentage by which to scale the image.

    Returns:
        numpy.ndarray: The resized image.

    This resizes the image proportionally based on the scale percentage, useful for 
    adjusting the output size for display or further processing.
    """
    def scale_image(self, image, scale_percent):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    """
    Applies a dithering effect to the image.

    Args:
        image (numpy.ndarray): The input grayscale image.
        pixel_distance (int, optional): The distance between pixels to be dithered. Defaults to 5.
        pixel_threshold (int, optional): The intensity threshold for applying dithering. Defaults to 128.
        diagonal (bool, optional): Whether to apply dithering in a diagonal pattern (currently unused). Defaults to True.

    Returns:
        numpy.ndarray: The dithered image.

    This method reduces the number of visible intensity levels in the image by applying a pattern 
    that simulates shading, mimicking the texture seen in manga art.
    """
    def dither(self, image, pixel_distance=5, pixel_threshold=128, diagonal=True):
        dithered_image = image.copy()
        rows, cols = image.shape
        
        for i in range(0, rows, pixel_distance):
            for j in range(0, cols, pixel_distance):
                if dithered_image[i][j] < pixel_threshold:

                    dithered_image[i][j] = 127

        return dithered_image

def main():
    root = ctk.CTk()
    app = MangaFilterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
