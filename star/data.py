from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from star.config import (
    DATA_DIR,
    LABELS_FILE,
    CLASS_NAMES,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_STATE,
)

LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}

RAW_GALAXY_ZOO_DIR = DATA_DIR / "raw" / "galaxy_zoo_2"
RAW_LABELS_FILE = RAW_GALAXY_ZOO_DIR / "gz2_hart16.csv"
RAW_MAPPING_FILE = RAW_GALAXY_ZOO_DIR / "gz2_filename_mapping.csv"
RAW_IMAGE_DIR = RAW_GALAXY_ZOO_DIR / "images"

SMOOTH_COLUMN = "t01_smooth_or_features_a01_smooth_debiased"
FEATURED_COLUMN = "t01_smooth_or_features_a02_features_or_disk_debiased"


def prepare_labels_file(threshold=0.8):
    hart_df = pd.read_csv(RAW_LABELS_FILE)
    mapping_df = pd.read_csv(RAW_MAPPING_FILE)

    hart_df = hart_df.rename(columns={"dr7objid": "objid"})

    dataframe = pd.merge(hart_df, mapping_df, on="objid", how="inner")

    elliptical_df = dataframe[dataframe[SMOOTH_COLUMN] >= threshold].copy()
    elliptical_df["label"] = "elliptical"

    spiral_df = dataframe[dataframe[FEATURED_COLUMN] >= threshold].copy()
    spiral_df["label"] = "spiral"

    dataframe = pd.concat([elliptical_df, spiral_df], ignore_index=True)
    dataframe = dataframe[["asset_id", "label"]].drop_duplicates(subset=["asset_id"])

    dataframe["image_name"] = dataframe["asset_id"].astype(str) + ".jpg"
    dataframe = dataframe[["image_name", "label"]]

    dataframe["image_exists"] = dataframe["image_name"].apply(
        lambda name: (RAW_IMAGE_DIR / name).exists()
    )
    dataframe = dataframe[dataframe["image_exists"]].copy()
    dataframe = dataframe.drop(columns=["image_exists"])

    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(LABELS_FILE, index=False)

    return dataframe


class GalaxyDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = self.image_dir / row["image_name"]
        image = Image.open(image_path).convert("RGB")
        label = LABEL_TO_INDEX[row["label"]]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    return train_transform, eval_transform


def load_labels():
    if not LABELS_FILE.exists():
        prepare_labels_file()

    dataframe = pd.read_csv(LABELS_FILE)
    dataframe = dataframe[dataframe["label"].isin(CLASS_NAMES)].copy()
    return dataframe


def split_data(dataframe):
    train_df, temp_df = train_test_split(
        dataframe,
        test_size=0.3,
        stratify=dataframe["label"],
        random_state=RANDOM_STATE,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE,
    )

    return train_df, val_df, test_df


def create_dataloaders():
    dataframe = load_labels()
    train_df, val_df, test_df = split_data(dataframe)
    train_transform, eval_transform = get_transforms()

    train_dataset = GalaxyDataset(train_df, RAW_IMAGE_DIR, train_transform)
    val_dataset = GalaxyDataset(val_df, RAW_IMAGE_DIR, eval_transform)
    test_dataset = GalaxyDataset(test_df, RAW_IMAGE_DIR, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader