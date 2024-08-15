import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import SimpleITK as sitk
from glob import glob
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from scipy.ndimage import zoom
import config
from models.NMSeg import SSTNet
from utils import logger


def predict_one_img(model, img_dataset, args):
    dataloader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=0, shuffle=False)
    model.eval()

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            output = model(data)
            img_dataset.update_result(output.cpu())

    pred = img_dataset.recompose_result()
    pred = torch.argmax(pred, dim=1).numpy().astype('uint8')

    if args.postprocess:
        pass  # 添加后处理逻辑（如果需要）

    return sitk.GetImageFromArray(np.squeeze(pred, axis=0))


class ImgDataset():
    def __init__(self, data_path, args):
        self.cut_size = args.test_cut_size
        self.cut_stride = args.test_cut_stride
        self.n_labels = args.n_labels

        self.img = sitk.ReadImage(data_path, sitk.sitkInt16)
        self.data_np = sitk.GetArrayFromImage(self.img)
        self.ori_shape = self.data_np.shape

        self.data_np = self._preprocess(self.data_np, args)
        self.resized_shape = self.data_np.shape

        self.data_np = self._pad_img(self.data_np, self.cut_size, self.cut_stride)
        self.padding_shape = self.data_np.shape

        self.data_np = self._extract_ordered_overlap(self.data_np, self.cut_size, self.cut_stride)
        self.result = None

    def _preprocess(self, img, args):
        img = zoom(img, (args.slice_down_scale, args.xy_down_scale, args.xy_down_scale), order=0)
        img = np.clip(img, args.lower, args.upper)
        return img / args.norm_factor

    def __getitem__(self, index):
        data = torch.FloatTensor(self.data_np[index]).unsqueeze(0)
        return data

    def __len__(self):
        return len(self.data_np)

    def update_result(self, tensor):
        self.result = tensor if self.result is None else torch.cat((self.result, tensor), dim=0)

    def recompose_result(self):
        patch_s = self.result.shape[2]
        N_patches_img = (self.padding_shape[0] - patch_s) // self.cut_stride + 1
        assert self.result.shape[0] == N_patches_img

        full_prob = torch.zeros((self.n_labels, self.padding_shape[0], self.ori_shape[1], self.ori_shape[2]))
        full_sum = torch.zeros_like(full_prob)

        for s in range(N_patches_img):
            slice_range = slice(s * self.cut_stride, s * self.cut_stride + patch_s)
            full_prob[:, slice_range] += self.result[s]
            full_sum[:, slice_range] += 1

        final_avg = full_prob / full_sum
        return final_avg[:, :self.ori_shape[0], :self.ori_shape[1], :self.ori_shape[2]].unsqueeze(0)

    def _pad_img(self, img, size, stride):
        img_s, img_h, img_w = img.shape
        pad_s = (stride - (img_s - size) % stride) % stride
        padded_img = np.pad(img, ((0, pad_s), (0, 0), (0, 0)), mode='constant', constant_values=0)
        return padded_img

    def _extract_ordered_overlap(self, img, size, stride):
        img_s, img_h, img_w = img.shape
        N_patches_img = (img_s - size) // stride + 1
        patches = np.array([img[s * stride: s * stride + size] for s in range(N_patches_img)], dtype=np.float32)
        return patches


def TestDatasets(dataset_path, args):
    data_list = sorted(glob(dataset_path))
    print(f"The number of test samples is: {len(data_list)}")
    for datapath in data_list:
        print(f"\nStart Evaluate: {datapath}")
        yield ImgDataset(datapath, args=args), datapath[:-4] + ".tif", datapath


if __name__ == '__main__':
    args = config.args
    test_data_path = "../*.tif"
    save_path = " "
    device = torch.device('cpu' if args.cpu else 'cuda')

    model = SSTNet(_conv_repr=True, _pe_type="learned").to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)
    ckpt = torch.load(' ')
    model.load_state_dict(ckpt['net'], strict=False)

    test_log = logger.Test_Logger(save_path, "test_log")
    result_save_path = os.path.join(save_path, 'result')
    os.makedirs(result_save_path, exist_ok=True)

    datasets = TestDatasets(test_data_path, args=args)
    zoom_factor = (1 / 1.0, 1.0, 1.0)

    for img_dataset, file_name, datapath in datasets:
        pred_img = predict_one_img(model, img_dataset, args)
        datanp_array = sitk.GetArrayFromImage(pred_img)
        labeled_array = label(datanp_array > 0.5)
        pred = remove_small_objects(labeled_array, min_size=15)

        soma_loc_swc = [
            [rid + 1, 0, int(z / zoom_factor[2]), int(y / zoom_factor[1]), int(x / zoom_factor[0]), 0, -1]
            for rid, r in enumerate(regionprops(pred))
            for z, y, x in [r.centroid]
        ]

        output_path = file_name[:-4] + '.swc'
        np.savetxt(output_path, np.array(soma_loc_swc).astype(int), fmt='%d')
        print(f"{os.path.basename(file_name)}: {len(soma_loc_swc)} somata")
