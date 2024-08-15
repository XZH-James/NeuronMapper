"""
The centroid location of the connected region was extracted from each image, saved as an SWC file, and the total number
of cell bodies detected in all images was counted. SWC format data is commonly used in the study of neuronal structure
"""

import numpy as np
import tiffile as tiff
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from glob import glob
import os


def results2swc(seg, min_size=20, zoom_factor=(1.0, 1.0, 1.0)):
    """
    Converts a segmentation mask to SWC format.

    Parameters:
        seg (numpy.ndarray): The segmented image.
        min_size (int): Minimum size for removing small objects.
        zoom_factor (tuple): The scaling factor for each axis.

    Returns:
        numpy.ndarray: An array in SWC format representing soma locations.
    """
    labeled_seg = label(seg > 0.5)
    cleaned_seg = remove_small_objects(labeled_seg, min_size=min_size)
    regions = regionprops(cleaned_seg)

    soma_loc_swc = []
    for rid, region in enumerate(regions):
        z, y, x = region.centroid
        soma_loc_swc.append([
            rid + 1, 0,
            int(z / zoom_factor[2]),
            int(y / zoom_factor[1]),
            int(x / zoom_factor[0]),
            0, -1
        ])

    return np.array(soma_loc_swc)


def process_images(image_paths, min_size=20, zoom_factor=(1.0, 1.0, 1.0)):
    """
    Process a list of image paths to extract soma locations and save them in SWC format.

    Parameters:
        image_paths (list): List of file paths to TIFF images.
        min_size (int): Minimum size for removing small objects.
        zoom_factor (tuple): The scaling factor for each axis.

    Returns:
        int: Total number of soma detected across all images.
    """
    total_soma_count = 0

    for image_path in image_paths:
        segresult_sample_img = tiff.imread(image_path)
        soma_loc_swc = results2swc(segresult_sample_img, min_size=min_size, zoom_factor=zoom_factor)
        total_soma_count += len(soma_loc_swc)

        output_path = f"{os.path.splitext(image_path)[0]}.swc"
        np.savetxt(output_path, soma_loc_swc.astype(int), fmt='%d')

        print(f"{os.path.basename(image_path)}: {len(soma_loc_swc)} somata")

    return total_soma_count


if __name__ == "__main__":
    # Set the directory or file pattern to search for TIFF images
    image_dir = '../.tif '  # Predict result path
    test_sample_list = glob(image_dir)

    # Process all images and count the total number of soma detected
    total_soma = process_images(test_sample_list)

    print(f"Total number of somata detected: {total_soma}")
