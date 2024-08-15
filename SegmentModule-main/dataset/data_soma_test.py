"""
A custom PyTorch dataset class ImgDataset for processing 3D medical image data and provides an iterative method,
TestDatasets, for loading and processing test datasets

"""

import os
import torch
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from torch.utils.data import Dataset
from glob import glob


class ImgDataset(Dataset):
    def __init__(self, data_path, label_path, args):
        self.n_labels = args.n_labels
        self.cut_size = args.test_cut_size
        self.cut_stride = args.test_cut_stride

        # 读取并预处理图像数据
        self.data_np = self.load_and_preprocess_image(data_path, args)
        self.padding_shape = self.data_np.shape
        self.data_np = self.extract_ordered_overlap(self.data_np, self.cut_size, self.cut_stride)

        # 读取并处理标签数据
        self.label_np = self.load_and_process_label(label_path, args.n_labels)

        # 预测结果保存
        self.result = None

    def __getitem__(self, index):
        data = torch.FloatTensor(self.data_np[index]).unsqueeze(0)
        return data

    def __len__(self):
        return len(self.data_np)

    def update_result(self, tensor):
        self.result = tensor if self.result is None else torch.cat((self.result, tensor), dim=0)

    def recompone_result(self):
        patch_s = self.result.shape[2]
        N_patches_img = (self.padding_shape[0] - patch_s) // self.cut_stride + 1

        full_prob = torch.zeros((self.n_labels, self.padding_shape[0], self.ori_shape[1], self.ori_shape[2]))
        full_sum = torch.zeros_like(full_prob)

        for s in range(N_patches_img):
            full_prob[:, s * self.cut_stride:s * self.cut_stride + patch_s] += self.result[s]
            full_sum[:, s * self.cut_stride:s * self.cut_stride + patch_s] += 1

        final_avg = full_prob / full_sum
        img = final_avg[:, :self.ori_shape[0], :self.ori_shape[1], :self.ori_shape[2]]
        return img.unsqueeze(0)

    def load_and_preprocess_image(self, path, args):
        img = sitk.ReadImage(path, sitk.sitkInt16)
        data_np = sitk.GetArrayFromImage(img)
        self.ori_shape = data_np.shape

        # Resize and normalize image
        data_np = ndimage.zoom(data_np, (args.slice_down_scale, args.xy_down_scale, args.xy_down_scale), order=3)
        data_np = np.clip(data_np, args.lower, args.upper)
        data_np /= args.norm_factor

        # Padding
        return self.padding_img(data_np, self.cut_size, self.cut_stride)

    def load_and_process_label(self, path, n_labels):
        seg = sitk.ReadImage(path, sitk.sitkInt8)
        label_np = sitk.GetArrayFromImage(seg)
        if n_labels == 2:
            label_np = (label_np > 0).astype(np.int8)
        return torch.from_numpy(np.expand_dims(label_np, axis=0)).long()

    def padding_img(self, img, size, stride):
        img_s, img_h, img_w = img.shape
        s = img_s + (stride - (img_s - size) % stride) if (img_s - size) % stride != 0 else img_s

        padded_img = np.zeros((s, img_h, img_w), dtype=np.float32)
        padded_img[:img_s] = img
        return padded_img

    def extract_ordered_overlap(self, img, size, stride):
        img_s, img_h, img_w = img.shape
        N_patches_img = (img_s - size) // stride + 1

        patches = np.array([
            img[s * stride: s * stride + size]
            for s in range(N_patches_img)
        ], dtype=np.float32)

        return patches


def TestDatasets(dataset_path, args):
    data_list = sorted(glob(os.path.join(dataset_path, 'image/*')))
    label_list = sorted(glob(os.path.join(dataset_path, 'label/*')))
    print(f"The number of test samples is: {len(data_list)}")
    for data_path, label_path in zip(data_list, label_list):
        print(f"\nStart Evaluate: {data_path}")
        yield ImgDataset(data_path, label_path, args=args), os.path.basename(data_path).split('-')[-1]


def to_one_hot_3d(tensor, n_classes=2):
    n, s, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, s, h, w).scatter_(1, tensor.view(n, 1, s, h, w), 1)
    return one_hot
