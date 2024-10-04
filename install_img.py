from pycocotools.coco import COCO
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os

# download annotations from https://cocodataset.org/#download

"""
To do:
Add a function to convert the image to 240x240 before saving it
"""

TOTAL_IMAGES = 1500
TESTING_PERCENTAGE = 0.1
TRAINING_DIRECTORTY = "./Dataset/Train"
TESTING_DIRECTORY = "./Dataset/Test"

# Create directories if they do not exist
os.makedirs(TRAINING_DIRECTORTY, exist_ok=True)
os.makedirs(TESTING_DIRECTORY, exist_ok=True)

coco = COCO("./annotations/instances_train2017.json")


def download_image(folder, im):
    """Downloads the image from the COCO dataset"""
    if os.path.exists(folder + im["file_name"]):
        # print(f"File {folder + im['file_name']} already exists. Skipping download.")
        return
    try:
        img_data = requests.get(im["coco_url"], timeout=10).content
        with open(folder + im["file_name"], "wb") as handler:
            handler.write(img_data)
    except Exception as e:
        print(f"Error downloading {im}: {e}")


def download_coco_images(label):
    """Downloads the images from the COCO dataset"""
    # instantiate COCO specifying the annotations json path

    # Specify a list of category names of interest
    catIds = coco.getCatIds(label)

    # Get the corresponding image ids and images using loadImgs
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)
    print(f"Number of images: {len(images)}")
    # the first image in the list
    # Save the images into a local folder
    no_of_training_images = int((1 - TESTING_PERCENTAGE) * TOTAL_IMAGES)
    training_images = images[0:no_of_training_images]
    testing_images = images[no_of_training_images:TOTAL_IMAGES]

    label_train_directory = f"{TRAINING_DIRECTORTY}/{label[0]}/"
    label_test_directory = f"{TESTING_DIRECTORY}/{label[0]}/"
    os.makedirs(label_train_directory, exist_ok=True)
    os.makedirs(label_test_directory, exist_ok=True)
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(
            tqdm(
                executor.map(
                    lambda im: download_image(label_train_directory, im),
                    training_images,
                ),
                total=len(training_images),
                desc="Downloading Training Images",
            )
        )

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(
            tqdm(
                executor.map(
                    lambda im: download_image(label_test_directory, im), testing_images
                ),
                total=len(testing_images),
                desc="Downloading Testing Images",
            )
        )


def downloder(labels):
    # for label in labels:
    #     print(f"Downloading images for {label}")
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(
            tqdm(
                executor.map(lambda label: download_coco_images([label]), labels),
                total=len(labels),
                desc="Labels Completed",
                position=0,
            )
        )


if __name__ == "__main__":
    downloder(labels=["laptop", "cat", "dog", "person", "chair", "bottle", "car"])
