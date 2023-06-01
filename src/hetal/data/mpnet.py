from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

file_dir = Path(__file__).resolve().parent

SNIPS_LABEL_TO_IDX = {
    "PlayMusic": 0,
    "GetWeather": 1,
    "BookRestaurant": 2,
    "AddToPlaylist": 3,
    "RateBook": 4,
    "SearchCreativeWork": 5,
    "SearchScreeningEvent": 6,
}


def download_raw_data():
    name = "snips"
    url = "https://raw.github.com/LeePleased/StackPropagation-SLU/master/data/snips/"
    data_dir = file_dir / name
    data_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "dev", "test"]:
        raw_filename = f"sst_{split}.txt" if name == "sst5" else f"{split}.txt"
        filename = f"{split}.txt"
        filepath = data_dir / filename
        response = requests.get(url + raw_filename)
        if not filepath.exists():
            with open(filepath, "w") as f:
                f.write(response.text)
            print(f"Downloaded {filename} to {filepath}.")
        else:
            print(f"{filename} already exists at {filepath}.")


def snips_reader(file_path: Path):
    texts, slots, intents = [], [], []
    text, slot = [], []
    with open(file_path, "r") as fr:
        for line in fr.readlines():
            items = line.strip().split()
            if len(items) == 1:
                texts.append(text)
                slots.append(slot)
                intents.append(items)
                text, slot = [], []
            elif len(items) == 2:
                text.append(items[0].strip())
                slot.append(items[1].strip())

    assert len(texts) == len(intents)
    for text, intent in zip(texts, intents):
        text = " ".join(text)
        assert len(intent) == 1
        intent = str(SNIPS_LABEL_TO_IDX[intent[0]])
        yield (intent, text)


def load_mpnet():
    model = SentenceTransformer("paraphrase-mpnet-base-v2")
    return model


if __name__ == "__main__":
    download_raw_data()
    model = load_mpnet()
    data_dir = file_dir / "snips"

    for split in ["train", "dev", "test"]:
        raw_file = data_dir / f"{split}.txt"

        features = []
        labels = []
        for item in tqdm(snips_reader(raw_file)):
            label, text = item
            label = int(label)
            mpnet_feature = model.encode(text).tolist()
            features.append(mpnet_feature)
            labels.append(label)

        features = np.array(features)
        labels = np.array(labels)
        if split == "dev":
            split = "val"
        np.save(data_dir / f"features_{split}.npy", features)
        np.save(data_dir / f"labels_{split}.npy", labels)
