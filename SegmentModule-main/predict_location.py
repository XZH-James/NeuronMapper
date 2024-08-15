"""
Our deep learning model (SSTNet) is used to automate the detection of cell bodies in 3D images, and the detected cell body
locations are saved in a.swc file format for subsequent analysis and processing

"""


import os
import numpy as np
import tiffile as tiff
from glob import glob
from scipy.ndimage import zoom
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import torch
import config
from models.NMSeg import SSTNet


def infer(img, model, stride=176, patch_size=192):
    img = zoom(img, (1.0, 1.0, 1.0), order=0)

    w, h, d = img.shape
    w_steps = (w + stride - 1) // stride
    h_steps = (h + stride - 1) // stride
    d_steps = (d + stride - 1) // stride

    op_size = (patch_size - stride) // 2
    pad_size = [(op_size, w_steps * stride - w + op_size),
                (op_size, h_steps * stride - h + op_size),
                (op_size, d_steps * stride - d + op_size)]

    img = np.pad(img, pad_size, mode='symmetric')
    img = img[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions

    predseg = []

    for ix in range(w_steps):
        for iy in range(h_steps):
            for iz in range(d_steps):
                xmin, xmax = ix * stride, ix * stride + patch_size
                ymin, ymax = iy * stride, iy * stride + patch_size
                zmin, zmax = iz * stride, iz * stride + patch_size

                patch = img[:, xmin:xmax, ymin:ymax, zmin:zmax, :]
                pred = model.predict(patch, batch_size=1)
                predseg.append(pred)

    seg = np.zeros((w_steps * stride, h_steps * stride, d_steps * stride))

    for block_id, pred in enumerate(predseg):
        iz = block_id % d_steps
        iy = (block_id // d_steps) % h_steps
        ix = block_id // (h_steps * d_steps)
        xmin, xmax = ix * stride, ix * stride + stride
        ymin, ymax = iy * stride, iy * stride + stride
        zmin, zmax = iz * stride, iz * stride + stride

        seg[xmin:xmax, ymin:ymax, zmin:zmax] = pred[0, op_size:-op_size, op_size:-op_size, op_size:-op_size, 0]

    return results2swc(seg[:w, :h, :d])


def results2swc(seg, zoom_factor=(1.0, 1.0, 1.0), min_size=20):
    seg = label(seg > 0.5)
    seg = remove_small_objects(seg, min_size=min_size)
    soma_loc_swc = []

    for rid, region in enumerate(regionprops(seg)):
        x, y, z = region.centroid
        soma_loc_swc.append([
            rid + 1, 0,
            int(z / zoom_factor[2]),
            int(y / zoom_factor[1]),
            int(x / zoom_factor[0]),
            0, -1
        ])

    return np.array(soma_loc_swc)


def main():
    args = config.args
    device = torch.device('cpu' if args.cpu else 'cuda')
    model = SSTNet(_conv_repr=True, _pe_type="learned").to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)

    net = torch.load(' ')
    model.load_state_dict(net['net'], strict=False)

    test_sample_list = glob(' ')

    for test_sample_path in test_sample_list:
        test_sample_img = tiff.imread(test_sample_path)
        soma_loc_swc = infer(test_sample_img, model)

        output_path = f"{test_sample_path[:-4]}.swc"
        np.savetxt(output_path, soma_loc_swc.astype(int), fmt='%d')
        print(f"{os.path.basename(test_sample_path)}: {len(soma_loc_swc)} somata")


if __name__ == "__main__":
    main()
