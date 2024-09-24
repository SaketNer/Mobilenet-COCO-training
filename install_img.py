from pycocotools.coco import COCO
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

#download annotations from https://cocodataset.org/#download

def download_image(im):
    """Downloads the image from the COCO dataset"""
    try:
        img_data = requests.get(im['coco_url'],timeout=10).content
        with open('./Dataset/coco_person/' + im['file_name'], 'wb') as handler:
            handler.write(img_data)
    except Exception as e:
        print(f"Error downloading {im}: {e}")


def download_coco_images(labels):
    """Downloads the images from the COCO dataset"""
    # instantiate COCO specifying the annotations json path
    coco = COCO('./annotations/instances_train2017.json')

    # Specify a list of category names of interest
    catIds = coco.getCatIds(labels)

    # Get the corresponding image ids and images using loadImgs
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)
    print(f"Number of images: {len(images)}")
    #the first image in the list
    # Save the images into a local folder
    images = images[0:10]
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(tqdm(executor.map(download_image, images), total=len(images), desc="Downloading Images"))

if __name__ == "__main__":
    download_coco_images(labels=['person'])
