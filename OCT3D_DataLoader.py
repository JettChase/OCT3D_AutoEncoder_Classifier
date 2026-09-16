#https://huggingface.co/datasets/harvardairobotics/Harvard-GF
import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

DATA = r"C:\Users\Mike\Desktop\PC_ONLY_3DOCT\data"

summary = pd.read_csv(DATA + r"\data_summary.csv").set_index("filename")
demographics = pd.get_dummies(summary.drop(columns=["glaucoma", "use", "language"]), dtype=float)
N_DEMOGRAPHICS = demographics.shape[1]


class OCTDataset(Dataset):
    def __init__(self, folder, root=DATA):
        self.files = glob.glob(root + f"\\{folder}\\*.npz")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        #cropping the optic nerve, normalization happens on the GPU
        v = torch.from_numpy(d['oct_bscans'][36:164, 36:164, 36:164]).unsqueeze(0)
        label = int(d['glaucoma'])

        filename = os.path.basename(self.files[i])
        demo = torch.tensor(demographics.loc[filename].values, dtype=torch.float32)

        return v, label, demo


def load_oct_loader(folder, batch_size, shuffle, num_workers=0, root=DATA):
    return DataLoader(
        OCTDataset(folder, root),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )