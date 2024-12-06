import os
import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox, PhotoImage
from PIL import Image, ImageTk

class MangaFilterApp:
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

    def toggle_threshold(self):
        self.is_threshold = not self.is_threshold
        
        self.threshold_switch.configure(
            text="On" if self.is_threshold else "Off"
        )
        
        if self.original_image is not None:
            self.process_image()

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

    def update_image(self, name, value, divisor):
        self.sliders[name]['value_label'].configure(text=str(int(value)))
        
        if self.original_image is not None:
            self.process_image()

    def toggle_dither(self):
        self.is_dithering = not self.is_dithering
        
        self.dither_switch.configure(
            text="On" if self.is_dithering else "Off"
        )
        
        if self.original_image is not None:
            self.process_image()

    def process_image(self):
        sigma = self.sliders['Sigma']['slider'].get() / 10.0
        k = self.sliders['k']['slider'].get() / 10.0
        sharpen = self.sliders['Sharpen']['slider'].get()
        epsilon = self.sliders['Epsilon']['slider'].get() / 100.0
        phi = self.sliders['Phi']['slider'].get()
        scale = self.sliders['Scale']['slider'].get()
        pixel_distance = self.sliders['Dither Dist.']['slider'].get()
        threshold_value = 128  # Set the threshold value here

        result = self.xdog(
            self.original_image, 
            sigma, k, sharpen, epsilon, phi
        )

        if self.is_threshold:
            result = self.apply_threshold(result, threshold_value)

        if self.is_dithering:
            result = self.dither(result, pixel_distance=int(pixel_distance))
        
        result = self.scale_image(result, scale)

        self.processed_image = result

        self.display_image(result)

    def apply_threshold(self, image, threshold_value):
        _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        return binary_image

    def display_image(self, cv_image):
        pil_image = Image.fromarray(cv_image)
        photo = ctk.CTkImage(light_image=pil_image, size=(pil_image.width, pil_image.height))
        
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def xdog(self, image, sigma_one, k, sharpen, epsilon, phi):
        rescaled = image.astype(np.float32) / 255.0
        img_a = cv2.GaussianBlur(rescaled, (0, 0), sigmaX=sigma_one)
        img_b = cv2.GaussianBlur(rescaled, (0, 0), sigmaX=sigma_one * k)
        scaled_diff = (sharpen + 1) * img_a - sharpen * img_b
        sharpened = rescaled * scaled_diff * 255
        mask = (rescaled * scaled_diff > epsilon).astype(np.float32)
        inverse_mask = 1.0 - mask
        soft_thresholded = 1.0 + np.tanh(phi * (sharpened / 255.0 - epsilon))
        result = (mask + inverse_mask * soft_thresholded) * 255.0
        return np.clip(result, 0, 255).astype(np.uint8)

    def scale_image(self, image, scale_percent):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

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
