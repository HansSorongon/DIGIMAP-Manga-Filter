import cv2
import numpy as np
from tkinter import Tk, filedialog

def soft_threshold(image, phi, epsilon):
    return np.tanh(phi * (image - epsilon))

def xdog(image, sigma_one, k, sharpen, epsilon, phi):
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

def scale_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def blend_dark_areas_with_hatch(image, hatch_path, percentiles=(10, 15, 50), alpha=2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hatch = cv2.imread(hatch_path, cv2.IMREAD_GRAYSCALE)
    hatch = cv2.resize(hatch, (image.shape[1], image.shape[0]))

    combined_mask = np.zeros_like(gray)
    kernel = np.ones((9, 9), np.uint8)
    
    for percentile in percentiles:
        threshold = np.percentile(gray, percentile)
        dark_mask = cv2.morphologyEx(
            cv2.morphologyEx((gray < threshold).astype(np.uint8) * 255, 
                             cv2.MORPH_OPEN, kernel), 
            cv2.MORPH_CLOSE, kernel
        )
        combined_mask = cv2.bitwise_or(combined_mask, dark_mask)

    hatch_3d = cv2.cvtColor(hatch, cv2.COLOR_GRAY2BGR)
    hatch_masked = cv2.bitwise_and(hatch_3d, image)
    return cv2.addWeighted(image, 1, hatch_masked, alpha, 0)

def dither(image, pixel_distance=5, pixel_threshold=128, diagonal=False):
    dithered_image = image.copy()
    rows, cols = image.shape
    
    for i in range(0, rows, pixel_distance):
        for j in range(0, cols, pixel_distance):
            if dithered_image[i][j] < pixel_threshold:
                dithered_image[i][j] = 127
                
                if diagonal and i + pixel_distance // 2 < rows and j + pixel_distance // 2 < cols:
                    dithered_image[i + pixel_distance // 2][j + pixel_distance // 2] = 127
    
    return dithered_image

def update_view(x):
    try:
        sigma = cv2.getTrackbarPos('Sigma', 'Manga Filter') / 10.0
        k = cv2.getTrackbarPos('k', 'Manga Filter') / 10.0
        sharpen = cv2.getTrackbarPos('Sharpen', 'Manga Filter')
        epsilon = cv2.getTrackbarPos('Epsilon', 'Manga Filter') / 100.0
        phi = cv2.getTrackbarPos('Phi', 'Manga Filter')
        scale = cv2.getTrackbarPos('Scale', 'Manga Filter')
        
        result = xdog(image, sigma, k, sharpen, epsilon, phi)

        # -- STYLIZE

        # result = cv2.GaussianBlur(result, (1, 1), 0)
        # _, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY)
        # result = dither(result, pixel_threshold = 128)

        result = scale_image(result, scale)
        
        cv2.imshow('Manga Filter', result)

    except cv2.error as e:
        print(f"Error in update_view: {e}")

def select_image():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")])

def main():
    filepath = select_image()
    if not filepath:
        print("No image selected. Exiting...")
        return

    global image
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to load the image.")
        return

    image = scale_image(image, 50)

    cv2.namedWindow('Manga Filter')
    cv2.createTrackbar('Scale', 'Manga Filter', 100, 100, update_view)
    cv2.createTrackbar('Sigma', 'Manga Filter', 3, 100, update_view)
    cv2.createTrackbar('k', 'Manga Filter', 19, 100, update_view)
    cv2.createTrackbar('Sharpen', 'Manga Filter', 19, 50, update_view)
    cv2.createTrackbar('Phi', 'Manga Filter', 10, 50, update_view)
    cv2.createTrackbar('Epsilon', 'Manga Filter', 15, 50, update_view)

    update_view(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
