"""
For 2D
A gray image is processed by distance transformation, region labeling and connected domain attribute analysis.
The final result includes the number of connected domains and the centroid position, and the processing results are
saved as images and PDF files

"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.ndimage import *
from skimage.measure import *
from skimage.morphology import *

soma_loc_swc = []
img = cv.imread('../.tif', cv.IMREAD_GRAYSCALE)  # input images path
kernel = np.ones((3, 3), np.uint8)
distance = cv.distanceTransform(img, cv.DIST_L2, 3)
seg = label(distance> 5, connectivity=None)

regions = regionprops(seg)
print("连通域个数：{}".format(len(regions)))
for rid, r in enumerate(regions):
    x, y = r.centroid
    print('x:{}'.format(x), "y:{}".format(y))
print('像素值：{}'.format(round(seg[51, 35], 2)))
cv.imwrite('D:\\test-2distance.tif', distance)

plt.subplot(141), plt.imshow(img, cmap='gray'), plt.title('org'), plt.axis('off')
plt.subplot(142), plt.imshow(seg, cmap='gray'), plt.title('seg'), plt.axis('off')
plt.subplot(143), plt.imshow(distance, cmap='gray'), plt.title('distance'), plt.axis('off')

plt.savefig('graph.pdf', bbox_inches='tight')
plt.show()
