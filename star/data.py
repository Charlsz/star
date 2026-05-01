from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from star.config import (
    LABELS_FILE,
    IMAGE_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_STATE,
)


LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}


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

    train_dataset = GalaxyDataset(train_df, IMAGE_DIR, train_transform)
    val_dataset = GalaxyDataset(val_df, IMAGE_DIR, eval_transform)
    test_dataset = GalaxyDataset(test_df, IMAGE_DIR, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader