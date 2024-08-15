import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import SimpleITK as sitk

INFO = "model/medmnist.json"


class MedMNIST(Dataset):
    flag = ...

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        ''' dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation
    
        '''

        self.root = root

        if download:
            self.download()

        path_list = self.root + '/list'
        name_list = split + "_path_list.txt"
        file_path = os.path.join(path_list, name_list)
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()
                if not lines:
                    break
                file_name_list.append(lines.split())
        myImg = np.array([])
        myLabel = np.array([])
        myFileName = np.array([])
        for file_name in file_name_list:
            label_name = int(file_name[0].split("\\")[-1][:-4].split('-')[-1])
            myLabel = np.append(myLabel, [label_name])
            myFileName = np.append(myFileName, [file_name])
            img = sitk.ReadImage(file_name, sitk.sitkInt16)  # Normal read training sets use this line
            # img = sitk.ReadImage(file_name[0][:-4].split('-')[0] + ".tif", sitk.sitkInt16)   # Use this line only for unlabeled tests
            img_array = sitk.GetArrayFromImage(img)
            img_array = np.squeeze(img_array, 0)
            myImg = np.append(myImg, img_array)

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.img = myImg
        self.label = myLabel
        self.myFileName = myFileName

    def __getitem__(self, index):
        img, target = self.img[index], self.label[index].astype(int)
        fileName = self.myFileName[index]
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, fileName

    def __len__(self):
        return self.img.shape[0]

    def __repr__(self):
        '''Adapted from torchvision.
        '''
        _repr_indent = 4
        head = "Dataset " + self.__class__.__name__

        body = ["Number of datapoints: {}".format(self.__len__()), "Root location: {}".format(self.root),
                "Split: {}".format(self.split), "Task: {}".format(self.info["task"]),
                "Number of channels: {}".format(self.info["n_channels"]),
                "Meaning of labels: {}".format(self.info["label"]),
                "Number of samples: {}".format(self.info["n_samples"]),
                "Description: {}".format(self.info["description"]), "License: {}".format(self.info["license"])]

        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)

    def download(self):
        try:
            from torchvision.datasets.utils import download_url
            download_url(url=self.info["url"], root=self.root,
                         filename="{}.npz".format(self.flag), md5=self.info["MD5"])
        except:
            raise RuntimeError('Something went wrong when downloading! ' +
                               'Go to the homepage to download manually. ' +
                               'https://github.com/MedMNIST/MedMNIST')


class somata(MedMNIST):
    flag = "somata"


