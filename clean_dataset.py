import os
import shutil



def delete_duplicates(base_path):
    folders = ['person', 'laptop', 'car', 'chair']
    seen_images = set()
    count = 0
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            continue

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if image_name in seen_images:
                os.remove(image_path)
                print(f"Deleted duplicate image: {image_path}")
                count += 1
            else:
                seen_images.add(image_name)
    print(f"Deleted {count} duplicate images.")

if __name__ == "__main__":
    base_path = './Dataset/Train'
    delete_duplicates(base_path)
    