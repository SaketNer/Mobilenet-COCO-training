import os
from PIL import Image
import keyboard
import shutil

# Path to the folder containing images
image_folder = "./Dataset/Train/Chair"

# Temporary trash folder to store deleted images
trash_folder = "./Dataset/Trash/Chair"
os.makedirs(trash_folder, exist_ok=True)

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

def display_image(image_path):
    """Display an image using PIL."""
    with Image.open(image_path) as img:
        img.show()

def delete_image(image_path):
    """Move the image to the trash folder."""
    print(f"Deleting {image_path}")
    shutil.move(image_path, trash_folder)

def main():
    idx = 0
    total_images = len(image_files)

    if total_images == 0:
        print("No images found.")
        return

    while idx < total_images:
        image_path = os.path.join(image_folder, image_files[idx])
        print(f"Displaying {image_files[idx]} ({idx + 1}/{total_images})")

        # Display the current image
        display_image(image_path)

        # Wait for user input (d to delete, space to go to the next image)
        while True:
            if keyboard.is_pressed('space'):
                print("Next image...")
                break
            elif keyboard.is_pressed('d'):
                delete_image(image_path)
                image_files.pop(idx)
                total_images -= 1
                break

        if idx >= total_images:
            break  # If no more images, exit loop
        idx += 1

if __name__ == "__main__":
    main()
