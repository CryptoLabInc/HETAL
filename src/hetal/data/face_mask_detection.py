import argparse
from pathlib import Path

import numpy as np
import torch
from bs4 import BeautifulSoup
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms import Resize
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTModel


file_dir = Path(__file__).resolve().parent
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def generate_box(obj):
    xmin = int(obj.find("xmin").text)
    ymin = int(obj.find("ymin").text)
    xmax = int(obj.find("xmax").text)
    ymax = int(obj.find("ymax").text)
    return [xmin, ymin, xmax, ymax]


def generate_label(obj):
    if obj.find("name").text == "with_mask":
        return 1
    elif obj.find("name").text == "mask_weared_incorrect":
        return 2
    elif obj.find("name").text == "without_mask":
        return 3
    else:
        raise ValueError()


def generate_target(image_id, file_path):
    with open(file_path) as f:
        data = f.read()
        soup = BeautifulSoup(data, "xml")
        objects = soup.find_all("object")

        num_objs = len(objects)

        # get bounding box coordinates for each mask
        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)  # xmin, ymin, xmax, ymax
        labels = torch.as_tensor(labels, dtype=torch.int64)

        img_id = torch.tensor([image_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target


if __name__ == "__main__":
    # Download datasets from https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
    # and put two folders 'annotations' and 'box-images' in the directory 'face-mask-detection'.

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="base")
    args = parser.parse_args()

    # load images and labels
    if args.model == "base":
        data_dir = Path("./face-mask-detection")
    else:
        data_dir = Path(f"./face-mask-detection_{args.model}")
    image_dir = data_dir / "images"
    label_dir = data_dir / "annotations"
    image_num = 853

    box_image_dir = data_dir / "box-images"
    box_image_dir.mkdir(exist_ok=True)
    box_cnt = 0
    box_feature_path = data_dir / "features.csv"
    box_label_path = data_dir / "labels.csv"
    box_label_list = []

    for i in tqdm(range(image_num)):
        image_path = image_dir / ("maksssksksss" + str(i) + ".png")
        label_path = label_dir / ("maksssksksss" + str(i) + ".xml")

        image = Image.open(image_path).convert("RGB")
        label = generate_target(i, label_path)

        boxes = label["boxes"]
        num_boxes = len(boxes)
        for j in range(num_boxes):
            xmin, ymin, xmax, ymax = boxes[j].type(torch.int).tolist()

            image_box = image.crop((xmin, ymin, xmax, ymax))
            image_box.save(box_image_dir / f"box{box_cnt}.png")
            # label["labels"][j] = 1 or 2 or 3 (with mask, mask worn incorrectly, without mask)
            label_box = label["labels"][j].item() - 1
            box_label_list.append(label_box)

            box_cnt += 1

    print("total boxes: ", box_cnt)  # 4072
    box_label_arr = np.array(box_label_list)

    if args.model == "base":
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    elif args.model == "large":
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-large-patch32-384")
        model = ViTModel.from_pretrained("google/vit-large-patch32-384")

    feature_arr = []
    sz = 224 if args.model in ["base", "large"] else 384
    transform = Resize(size=(sz, sz))
    for i in tqdm(range(box_cnt)):
        img_path = box_image_dir / f"box{i}.png"
        img = Image.open(img_path)
        img = transform(img)

        feature = np.array(feature_extractor(img)["pixel_values"])  # (1, 3, 224, 224)
        feature = torch.Tensor(feature)
        with torch.no_grad():
            output = model(feature)
        # hidden dimension: base 768, large 1024, huge 1280
        vit_feature = output.last_hidden_state[0, 0].tolist()
        feature_arr.append(vit_feature)

    columns = list(map(str, range(768)))
    feature_arr = np.array(feature_arr)

    # split as train, val, test with 7:1:2 ratio
    X_train, X_test, y_train, y_test = train_test_split(feature_arr, box_label_arr, test_size=0.2, stratify=box_label_arr, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, stratify=y_train, random_state=0)
    print("train, val, test: ", len(X_train), len(X_val), len(X_test))
    np.save(data_dir / "features_train.npy", X_train)
    np.save(data_dir / "features_val.npy", X_val)
    np.save(data_dir / "features_test.npy", X_test)
    np.save(data_dir / "labels_train.npy", y_train)
    np.save(data_dir / "labels_val.npy", y_val)
    np.save(data_dir / "labels_test.npy", y_test)
