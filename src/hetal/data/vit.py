import argparse
from pathlib import Path

import medmnist
import numpy as np
import torch
import torchvision
from medmnist import INFO
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm
from transformers import ViTModel

file_dir = Path(__file__).resolve().parent
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def load_dataloader(data: str, model_checkpoint: str):
    if model_checkpoint == "base":
        data_dir = file_dir / data
    else:
        data_dir = file_dir / f"{data}_{model_checkpoint}"

    mean = [0.5] if data in ["mnist"] else [0.5] * 3
    std = [0.5] if data in ["mnist"] else [0.5] * 3
    sz = 224 if model_checkpoint in ["base", "huge"] else 384
    transform = Compose(
        [
            Resize(size=(sz, sz)),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )
    batch_size = 64
    download = True

    if data == "mnist":
        train_dataset = torchvision.datasets.MNIST(data_dir, train=True, transform=transform, download=download)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.875, 0.125], generator=torch.Generator().manual_seed(0))
        test_dataset = torchvision.datasets.MNIST(data_dir, train=False, transform=transform, download=download)
    elif data == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, transform=transform, download=download)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.875, 0.125], generator=torch.Generator().manual_seed(0))
        test_dataset = torchvision.datasets.CIFAR10(data_dir, train=False, transform=transform, download=download)
    elif data == "dermamnist":
        info = INFO["dermamnist"]
        download = True
        DataClass = getattr(medmnist, info["python_class"])
        train_dataset = DataClass(split="train", transform=transform, download=download)        
        val_dataset = DataClass(split="val", transform=transform, download=download)        
        test_dataset = DataClass(split="test", transform=transform, download=download)        
    else:
        raise ValueError(f"Invalid dataset: {data}. Available datasets: 'mnist', 'cifar10', 'dermamnist'.")

    print(f"Dataset: {data}, Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, data_dir


def load_vit(checkpoint):
    if checkpoint == "base":
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        model.to(device)
    elif checkpoint == "large":
        model = ViTModel.from_pretrained("google/vit-large-patch32-384")
        model.to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--model", type=str, default="base")
    args = parser.parse_args()

    train_loader, val_loader, test_loader, data_dir = load_dataloader(args.data, args.model)
    data_dir.mkdir(exist_ok=True)
    model = load_vit(args.model)

    for split in ["train", "val", "test"]:
        features = []
        labels = []
        if split == "train":
            loader = train_loader
        elif split == "val":
            loader = val_loader
        else:
            loader = test_loader
        print(split)
        for input, label in tqdm(loader):
            input = input.to(device)
            if args.data in ["mnist"]:  # greyscale image
                input = input.expand(-1, 3, -1, -1)
            with torch.no_grad():
                output = model(input)
            vit_feature = output.last_hidden_state[:, 0, :].tolist()
            features += vit_feature

            label = torch.squeeze(label)
            labels += label.tolist()

        features = np.array(features)
        labels = np.array(labels)
        np.save(data_dir / f"features_{split}.npy", features)
        np.save(data_dir / f"labels_{split}.npy", labels)
