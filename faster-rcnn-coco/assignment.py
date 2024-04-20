import os
import requests
import zipfile

COCO = {
    "train2017": 
    {
        "images": "http://images.cocodataset.org/zips/train2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    },
    "val2017":
    {
        "images": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    },
    "test2017":
    {
        "images": "http://images.cocodataset.org/zips/test2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/image_info_test2017.zip",
    },
    "unlabeled2017":
    {
        "images": "http://images.cocodataset.org/zips/unlabeled2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip",
    },

}


def download_coco_dataset(dataset, savepath, split="train2017"):
    """
    Download COCO dataset images and annotations
    
    Args:
        dataset: dict, COCO dataset dictionary
        savepath: str, path to save the dataset, e.g. "data/coco"
        split: str, dataset split to download, one of ["train2017", "val2017", "test2017", "unlabeled2017"]
    """

    savepath = os.path.join(savepath, split)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    assert split in dataset, f"Split {split} not found in dataset."
    
    key = split
    
    if not os.path.exists(os.path.join(savepath, f"{key}.zip")):
        print(f"Downloading {key} images")
        response = requests.get(dataset[key]["images"])
        with open(f"{savepath}/{key}.zip", "wb") as file:
            file.write(response.content)
        with zipfile.ZipFile(f"{savepath}/{key}.zip", "r") as zip_ref:
            zip_ref.extractall(savepath)
    

    if not os.path.exists(os.path.join(savepath, f"{key}_annotations.zip")):            
        print(f"Downloading {key} annotations")
        response = requests.get(dataset[key]["annotations"])
        with open(f"{savepath}/{key}_annotations.zip", "wb") as file:
            file.write(response.content)
        with zipfile.ZipFile(f"{savepath}/{key}_annotations.zip", "r") as zip_ref:
            zip_ref.extractall(savepath)


# Download COCO dataset
# download_coco_dataset(COCO, "data/coco", "val2017")

# Display 10 images with annotations randomly

import random
import matplotlib.pyplot as plt

def display_images(images, annotations, num_images=10):
    """
    Display images with annotations
    
    Args:
        images: list, list of image filenames
        annotations: list, list of annotation filenames
        num_images: int, number of images to display
    """
    
    fig, axes = plt.subplots(2, num_images // 2, figsize=(20, 10))
    axes = axes.flatten()
    for i in range(num_images):
        idx = random.randint(0, len(images) - 1)
        img = plt.imread(images[idx])
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(images[idx].split("/")[-1])
    plt.tight_layout()
    plt.show()

import json

def load_annotations(annotations):
    """
    Load COCO annotations
    
    Args:
        annotations: str, path to COCO annotations file
    """
    
    with open(annotations, "r") as file:
        data = json.load(file)

    return data

def extract_annotations(annotations, image_id):
    """
    Extract annotations for a given image id
    
    Args:
        annotations: dict, COCO annotations
        image_id: int, image id
    """
    
    return [ann for ann in annotations["annotations"] if ann["image_id"] == image_id]

root = "data/coco/val2017"
img_root = os.path.join(root, "val2017")
ann_root = os.path.join(root, "annotations")
images = random.sample([os.path.join(img_root, file) for file in os.listdir(img_root)], 10)

annotations = load_annotations(os.path.join(ann_root, "instances_val2017.json"))

def read_classes(annotations):
    """
    Read class names from COCO annotations
    
    Args:
        annotations: dict, COCO annotations
    """
    
    return {cat["id"]: cat["name"] for cat in annotations["categories"]}


def draw_bboxes(img, bboxes, labels):
    """
    Draw bounding boxes on an image
    
    Args:
        img: ndarray, input image
        bboxes: list, list of bounding boxes
        labels: list, list of labels
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        label = labels[i]
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y-2, label, fontsize=12, color="r")
    
    plt.axis("off")
    plt.show()

classes = read_classes(annotations)
print(f"Classes: {classes}")

for img in images:
    img_id = int(img.split(os.sep)[-1].split(".")[0].split("_")[-1])
    anns = extract_annotations(annotations, img_id)
    print(f"Image: {img}")
    print(f"Annotations: {anns}")
    bboxes = [ann["bbox"] for ann in anns]
    labels = [classes[ann["category_id"]] for ann in anns]
    draw_bboxes(plt.imread(img), bboxes, labels)

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

class COCODatset(Dataset):
    def __init__(self, root, split="train2017", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        
        self.images = [os.path.join(os.path.join(root, split), file) for file in os.listdir(os.path.join(root, split))]
        self.annotations = load_annotations(os.path.join(root, "annotations", f"instances_{split}.json"))   
        self.classes = read_classes(self.annotations)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = plt.imread(self.images[idx])
        img_id = int(self.images[idx].split(os.sep)[-1].split(".")[0].split("_")[-1])
        anns = extract_annotations(self.annotations, img_id)
        
        bboxes = [ann["bbox"] for ann in anns]
        labels = [ann["category_id"] for ann in anns]
        
        if self.transform:
            img = self.transform(img)
        
        target = {}
        target["boxes"] = torch.tensor(bboxes, dtype=torch.float32)
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        
        return img, target
    

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])

dataset = COCODatset("data/coco", "val2017", transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for img, target in dataloader:
    


