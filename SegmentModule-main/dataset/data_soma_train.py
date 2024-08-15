"""
A custom dataset class for loading and processing 3D medical image data uses SimpleITK to read and preprocess medical
images, along with PyTorch's data loading mechanism and custom data enhancement operations to prepare data for model
training.
This code is called in train.py
"""
import os
import sys
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomCrop
from .transforms import RandomFlip_LR, RandomFlip_UD, Compose


def load_file_name_list(file_path):
    with open(file_path, 'r') as file_to_read:
        file_name_list = [line.strip().split() for line in file_to_read if line.strip()]
    return file_name_list


class TrainDataset(Dataset):
    def __init__(self, args):
        self.filename_list = load_file_name_list(os.path.join(args.dataset_path, 'train_path_list.txt'))
        self.crop_size = args.crop_size
        self.norm_factor = args.norm_factor
        self.transforms = Compose([
            RandomCrop(self.crop_size),
            RandomFlip_LR(prob=0.5),
            RandomFlip_UD(prob=0.5),
        ])

    def __getitem__(self, index):
        img_path, seg_path = self.filename_list[index]
        img = self.load_image(img_path, sitk.sitkInt16)
        seg = self.load_image(seg_path, sitk.sitkUInt8)

        img = self.preprocess_image(img)
        seg = torch.FloatTensor(seg).unsqueeze(0)

        if self.transforms:
            img, seg = self.transforms(img, seg)

        return img, seg.squeeze(0)

    def __len__(self):
        return len(self.filename_list)

    def load_image(self, file_path, image_type):
        img = sitk.ReadImage(file_path, image_type)
        return sitk.GetArrayFromImage(img)

    def preprocess_image(self, img_array):
        img_array = img_array / self.norm_factor
        return torch.FloatTensor(img_array.astype(np.float32)).unsqueeze(0)


if __name__ == "__main__":
    sys.path.append('')  # Add the path to import the configuration file config
    from config import args

    train_ds = TrainDataset(args)
    train_dl = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=1)

    for i, (img, seg) in enumerate(train_dl):
        print(f"Batch {i}: Image size = {img.size()}, Segmentation size = {seg.size()}")
