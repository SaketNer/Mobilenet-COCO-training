import numpy as np
import time

def rgb888_to_rgb565(img_array):
    # Ensure the image has the correct shape (1, height, width, channels)
    start_time = time.time()
    if img_array.ndim != 4 or img_array.shape[3] != 3:
        raise ValueError("Input image array must have shape (1, height, width, 3)")

    # Convert the RGB values from 8-bit to 16-bit
    r = (img_array[0, :, :, 0] >> 3) & 0x1F  # 5 bits for Red
    g = (img_array[0, :, :, 1] >> 2) & 0x3F  # 6 bits for Green
    b = (img_array[0, :, :, 2] >> 3) & 0x1F  # 5 bits for Blue

    # Combine the channels into a single 16-bit image
    #rgb565 = (r << 11) | (g << 5) | b
    rgb565 = np.stack([r, g, b], axis=-1) 
    print(f"Conversion took {time.time() - start_time:.2f} seconds")
    return rgb565