"""
For 3D
Performs basic image processing and analysis of 3D medical images, in particular identifying and extracting specific
regions in the image (such as biological structures of interest) through distance transformations and connected region
labeling, and saves the analysis results

"""


import numpy as np
import cv2 as cv
from scipy.ndimage import *
from skimage.measure import *
from skimage.morphology import *
import tiffile as tiff

soma_loc_swc = []
img = tiff.imread('../.tif')  # input images path
kernel = np.ones((3, 3, 3), np.uint8)
distance = cv.distanceTransform(img, cv.DIST_L2, 3)
seg = label(distance > 0.5,connectivity=None)
regions = regionprops(seg)

print("连通域个数：{}".format(len(regions)))

for rid, r in enumerate(regions):
    x, y, z = r.centroid
    print('x:{}'.format(x), "y:{}".format(y), "z:{}".format(z))

cv.imwrite('D:\\test-2distance.tif', distance)
