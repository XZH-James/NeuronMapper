"""
Preprocess 3D medical image data, especially resampling, cropping, expanding and other operations on images and
segmentation labels, and then save the processed data to a specified directory. Finally, the code generates dataset
index files for training, validation, and testing.
"""

import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage
from os.path import join
import config

class Soma_preprocess:
    def __init__(self, raw_dataset_path,fixed_dataset_path, args):
        self.raw_root_path = raw_dataset_path
        self.fixed_path = fixed_dataset_path
        self.classes = args.n_labels
        self.upper = args.upper
        self.lower = args.lower
        self.expand_slice = args.expand_slice
        self.size = args.min_slices
        self.scale = args.xy_down_scale
        self.xy_down_scale = self.scale
        self.slice_down_scale = args.slice_down_scale
        self.valid_rate = args.valid_rate

    def process1(self, img_path, seg_path, classes=None):
        img = sitk.ReadImage(img_path, sitk.sitkInt16)
        img_array = sitk.GetArrayFromImage(img)
        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        print("Ori shape:", img_array.shape, seg_array.shape)
        if classes == 2:
            seg_array[seg_array > 0] = 1

        img_array = ndimage.zoom(img_array,
                                (img.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale),
                                order=3)
        seg_array = ndimage.zoom(seg_array,
                                 (img.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale),
                                 order=0)

        # Find the slice at the beginning and end of the cell region and expand outward in each direction
        z = np.any(img_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]

        print("Cut out range:", str(start_slice) + '--' + str(end_slice))
        # If the number of remaining slices is insufficient, give up directly, such data is very small
        if end_slice - start_slice + 1 < self.size:
            print('Too little slice，give up the sample:', img_file)
            return None, None
        # Intercept reserved areas
        img_array = img_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]
        print("Preprocessed shape:", img_array.shape, seg_array.shape)
        # Save in the corresponding format
        new_img = sitk.GetImageFromArray(img_array)
        new_img.SetDirection(img.GetDirection())
        new_img.SetOrigin(img.GetOrigin())
        new_img.SetSpacing((img.GetSpacing()[0] * int(1 / self.xy_down_scale),
                            img.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))

        new_seg = sitk.GetImageFromArray(seg_array)
        new_seg.SetDirection(img.GetDirection())
        new_seg.SetOrigin(img.GetOrigin())
        new_seg.SetSpacing((img.GetSpacing()[0] * int(1 / self.xy_down_scale),
                            img.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))
        return new_img, new_seg

    def fix_data(self):
        if not os.path.exists(self.fixed_path):
            os.makedirs(join(self.fixed_path, 'image'))
            os.makedirs(join(self.fixed_path, 'label'))
        file_list = os.listdir(join(self.raw_root_path, 'image'))
        Numbers = len(file_list)
        print('Total numbers of samples is :', Numbers)

        global img_file
        for img_file, i in zip(file_list, range(Numbers)):
            print("==== {} | {}/{} ====".format(img_file, i + 1, Numbers))
            img_path = os.path.join(self.raw_root_path, 'image', img_file)
            seg_path = os.path.join(self.raw_root_path, 'label', img_file.replace('volume', 'label'))
            new_img, new_seg = self.process(img_path, seg_path, classes=self.classes)
            if new_img != None and new_seg != None:
                sitk.WriteImage(new_img, os.path.join(self.fixed_path, 'image', img_file))
                sitk.WriteImage(new_seg, os.path.join(self.fixed_path, 'label',
                                                      img_file.replace('volume', 'label').replace('.tif', '.tif')))


    def write_test_name_list(self):
        data_name_list = os.listdir(join(self.fixed_path, "image"))
        data_num = len(data_name_list)
        print('the fixed dataset total numbers of samples is :', data_num)
        random.shuffle(data_name_list)

        assert self.valid_rate < 1.0
        train_name_list = data_name_list[0:int(data_num * (1 - self.valid_rate))]
        val_name_list = data_name_list[
                        int(data_num * (1 - self.valid_rate)):int(data_num * ((1 - self.valid_rate) + self.valid_rate))]

        self.write_name_list(train_name_list, "train_path_list.txt")
        self.write_name_list(val_name_list, "val_path_list.txt")


    def write_train_val_name_list(self):
        data_name_list = os.listdir(join(self.fixed_path, "image"))
        data_num = len(data_name_list)
        print('the fixed dataset total numbers of samples is :', data_num)
        random.shuffle(data_name_list)

        assert self.valid_rate < 1.0
        train_name_list = data_name_list[0:int(data_num * (1 - self.valid_rate))]
        val_name_list = data_name_list[
                        int(data_num * (1 - self.valid_rate)):int(data_num * ((1 - self.valid_rate) + self.valid_rate))]

        self.write_name_list(train_name_list, "train_path_list.txt")
        self.write_name_list(val_name_list, "val_path_list.txt")

    def write_train_val_test_name_list(self, rate_a, rate_b):
        data_name_list = os.listdir(join(self.fixed_path, "image"))
        data_num = len(data_name_list)
        print('the fixed dataset total numbers of samples is :', data_num)
        random.shuffle(data_name_list)

        train_name_list = data_name_list[0:int(data_num * rate_a)]
        val_name_list = data_name_list[int(data_num * rate_a)+1:int(data_num * rate_b)]
        test_name_list = data_name_list[int(data_num * rate_b)+1:]

        self.write_name_list(train_name_list, "train_path_list.txt")
        self.write_name_list(val_name_list, "val_path_list.txt")
        self.write_name_list(test_name_list, "test_path_list.txt")

    def write_name_list(self, name_list, file_name):
        f = open(join(self.fixed_path, file_name), 'w')
        for name in name_list:
            img_path = os.path.join(self.fixed_path, 'image', name)
            seg_path = os.path.join(self.fixed_path, 'label', name.replace('volume', 'label'))
            f.write(img_path + ' ' + seg_path + "\n")
        f.close()

if __name__ == '__main__':

    raw_dataset_path = '../rawdataset'
    fixed_dataset_path = '../fixdataset'
    args = config.args
    tool = Soma_preprocess(raw_dataset_path, fixed_dataset_path, args)
    tool.fix_data()  # Trim and save the original image
    tool.write_train_val_name_list()  # Create an index txt file
