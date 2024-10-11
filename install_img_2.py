from pycocotools.coco import COCO
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os

MAX_IMAGES_PER_LABEL = 900
TESTING_PERCENTAGE = 0.05
TRAINING_DIRECTORTY = "./Dataset/Train"
TESTING_DIRECTORY = "./Dataset/Test"

# Create directories if they do not exist
os.makedirs(TRAINING_DIRECTORTY, exist_ok=True)
os.makedirs(TESTING_DIRECTORY, exist_ok=True)

coco = COCO("./annotations/instances_train2017.json")
downloaded_image_set = set()

def unique_id_per_class(labels):
    """Get the unique image ids for each class"""
    label_id_sets = []

    for label in labels:
        catIds = coco.getCatIds(label)
        imgIds = coco.getImgIds(catIds=catIds)
        imgIds_set = set(imgIds)
        label_id_sets.append(imgIds_set)

    overall_id_set = set.union(*label_id_sets)
    print(f"Number of images that have all the labels: {len(overall_id_set)}")

    for img_id in overall_id_set:
        lst = [i for i, label_set in enumerate(label_id_sets) if img_id in label_set]
        if len(lst) > 1:
            for i in lst:
                label_id_sets[i].remove(img_id)
    return label_id_sets

def download_image(label, im):
    """Downloads the image from the COCO dataset"""
    global downloaded_image_set

    folder_path = f"{TRAINING_DIRECTORTY}/{label}/"
    os.makedirs(folder_path, exist_ok=True)

    if im["file_name"] in downloaded_image_set:
        return
    downloaded_image_set.add(im["file_name"])
    
    if os.path.exists(folder_path + im["file_name"]) :
        # print(f"File {folder + im['file_name']} already exists or already downloaded. Skipping download.")
        return
    try:
        img_data = requests.get(im["coco_url"], timeout=10).content
        with open(folder_path + im["file_name"], "wb") as handler:
            handler.write(img_data)
            
    except Exception as e:
        print(f"Error downloading {im}: {e}")

def download_coco_images(labels, label_id_set, ignore_labels):
    
    for i, label_ids in enumerate(label_id_set):
        if(labels[i] in ignore_labels):
            print("Ignoring label:", labels[i])
            continue
        label_img_cnt = 0
        label_img_data = images = coco.loadImgs(label_ids)
        print("\nStarting lable :", labels[i], " with size: ", len(label_img_data), "\n")
        for id in tqdm(label_img_data):
            try:
                download_image(labels[i], id)
                label_img_cnt+=1
            except KeyboardInterrupt:
                print("Download interrupted by user")
                return
            except:
                print("Error downloading image")
            if(label_img_cnt>MAX_IMAGES_PER_LABEL):
                break

if __name__ == "__main__":
    #downloder(labels=["laptop", "person", "car","chair"])
    # List all available labels in COCO dataset
    cats = coco.loadCats(coco.getCatIds())
    all_labels = [cat['name'] for cat in cats]
    print("Available labels in COCO dataset:", all_labels)
    download_labels=["apple","car","potted plant","chair","person","dog","cat"]
    ignore_labels = ["dog","cat"]
    label_id_set = unique_id_per_class(download_labels)
    # Split the labels and label_id_set into three parts
    split_index_1 = len(download_labels) // 3
    split_index_2 = 2 * len(download_labels) // 3

    labels_part1 = download_labels[:split_index_1]
    labels_part2 = download_labels[split_index_1:split_index_2]
    labels_part3 = download_labels[split_index_2:]

    label_id_set_part1 = label_id_set[:split_index_1]
    label_id_set_part2 = label_id_set[split_index_1:split_index_2]
    label_id_set_part3 = label_id_set[split_index_2:]

    # Create a thread pool and download images in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.submit(download_coco_images, labels_part1, label_id_set_part1, ignore_labels)
        executor.submit(download_coco_images, labels_part2, label_id_set_part2, ignore_labels)
        executor.submit(download_coco_images, labels_part3, label_id_set_part3, ignore_labels)
    download_coco_images(download_labels, label_id_set, ignore_labels)
    