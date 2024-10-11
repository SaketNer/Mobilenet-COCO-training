import os
from PIL import Image, ImageEnhance, ImageOps
import random


def reduce_brightness(image_path, output_path, factor=0.5):

    image_quality = random.randint(50, 100)
    image_brightness = random.uniform(0.5, 1)

    image = Image.open(image_path)
    
    resized_img = ImageOps.fit(image, (240, 240), method=Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Brightness(resized_img)
    
    image_enhanced = enhancer.enhance(image_brightness)
    image_enhanced.save(output_path,quality=image_quality)

def process_images(input_dir, output_dir, factor=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                
                output_folder = os.path.dirname(output_path)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                
                reduce_brightness(input_path, output_path, factor)

if __name__ == "__main__":
    input_directory = './Dataset/Train'
    output_directory = './Dataset/Augment'
    brightness_factor = 0.5  # Adjust the factor as needed

    process_images(input_directory, output_directory, brightness_factor)