import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class TBDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        if row["label"] == 1:
            folder = "TB"
        else:
            folder = "Normal"

        img_path = os.path.join(
            self.image_dir,
            folder,
            row["image_name"]
        )

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing file: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        clinical = torch.tensor([
            row["age"],
            row["fever"],
            row["cough"],
            row["weight_loss"]
        ], dtype=torch.float32)

        label = torch.tensor(row["label"], dtype=torch.long)

        return image, clinical, label