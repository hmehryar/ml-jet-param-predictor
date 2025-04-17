# data/loader.py
import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

class JetDataset(Dataset):
    def __init__(self, file_list, global_max):
        self.samples = file_list
        self.global_max = global_max

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = np.load(sample['image_path']) / self.global_max
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, 32, 32)

        labels = {
            "energy_loss_output": torch.tensor(sample['energy_loss_output'], dtype=torch.float32),
            "alpha_output": torch.tensor(sample['alpha_output'], dtype=torch.long),
            "q0_output": torch.tensor(sample['q0_output'], dtype=torch.long),
        }
        return image, labels


def load_split_from_csv(csv_file, root_dir):
    df = pd.read_csv(csv_file)
    samples = []
    for _, row in df.iterrows():
        samples.append({
            'image_path': os.path.join(root_dir, row['relative_path']),
            'energy_loss_output': row['energy_loss_output'],
            'alpha_output': row['alpha_output'],
            'q0_output': row['q0_output'],
        })
    return samples


def get_dataloaders(cfg):
    train_list = load_split_from_csv(cfg["train_csv"], cfg["root_dir"])
    val_list = load_split_from_csv(cfg["val_csv"], cfg["root_dir"])

    train_ds = JetDataset(train_list, global_max=121.79151153564453)
    val_ds = JetDataset(val_list, global_max=121.79151153564453)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    return train_loader, val_loader
