"""
The main purpose of this code is to evaluate the performance of the image segmentation algorithm. By comparing the
predicted results to the actual label, the code is able to calculate multiple evaluation metrics (accuracy, recall,
accuracy, and F1 score). These metrics can help understand how the segmentation algorithm performs on the test set and
provide directions for improving the model.

To be specific:
Centroid extraction: Through connected component analysis, the centroid of each segment is extracted as the target
location of interest.
Match evaluation: By comparing the distance of the centroid position, the match between the prediction and the label is
calculated to obtain a statistical indicator.
Performance evaluation: The overall performance of the segmentation model is evaluated by multiple indicators, which is
suitable for the performance verification of automatic segmentation tasks in medical image analysis.

"""

import numpy as np
import SimpleITK as sitk
import os
from skimage.measure import *
from skimage.morphology import *


def print_directories(path):
    cnt = 0
    for name in os.listdir(path):
        cnt += 1

    return cnt


def precision(tp, fp, fn):
    return tp / (tp + fp) if tp > 0 else 0


def recall(tp, fp, fn):
    return tp / (tp + fn) if tp > 0 else 0


def accuracy(tp, fp, fn):
    return tp / (tp + fp + fn) if tp > 0 else 0


def f1(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0


def calCenterPoints(path):

    image1 = sitk.ReadImage(path)

    print("image Size(): ", image1.GetSize())

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)

    output_mask = cc_filter.Execute(image1)

    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_mask)

    num_connected_label = cc_filter.GetObjectCount()

    print("Number of connected fields: ", num_connected_label)

    data = []
    zoom_factor = (1 / 1.0, 1.0, 1.00)
    for i in range(1, num_connected_label + 1):

        tmp = [int(lss_filter.GetCentroid(i)[0]/zoom_factor[2]), int(lss_filter.GetCentroid(i)[1]/zoom_factor[1]), int(lss_filter.GetCentroid(i)[2]/zoom_factor[0])]
        data.append(tmp)

    data=np.array(data)

    return data


def results2swc(path):
    seg = sitk.ReadImage(path)

    print("image Size(): ", seg.GetSize())
    print(os.path.basename(path))
    zoom_factor = (1/1.0, 1.0, 1.0)
    datanp_array = sitk.GetArrayFromImage(seg)
    datanp_array = label(datanp_array > 0.5)
    datanp_array = remove_small_objects(datanp_array, min_size=15)
    regions = regionprops(datanp_array)
    soma_loc_swc = []
    for rid, r in enumerate(regions):
        x, y, z = r.centroid
        soma_loc_swc.append([(rid + 1), 0, int(z/zoom_factor[2]), int(y/zoom_factor[1]), int(x/zoom_factor[0]), 0, -1])
    return np.array(soma_loc_swc)

def calTpFpFn(ori, pre, diff):

    tp = 0
    cnt_ori = ori.shape[0]
    cnt_pre = pre.shape[0]
    for vec1 in ori:
        for vec2 in pre:
            distance = np.linalg.norm(vec1 - vec2)
            if distance <= diff:
                tp += 1
                break
    return {
        'tp': tp,
        'fp': cnt_ori - tp,
        'fn': cnt_pre - tp,
    }


def claMetrics(tp, fp, fn):
    return {
        'precision': precision(tp, fp, fn),
        'recall': recall(tp, fp, fn),
        'accuracy': accuracy(tp, fp, fn),
        'f1': f1(tp, fp, fn)
    }


if __name__ == "__main__":

    path_1 = " "  # Ground Truth label path
    path_2 = " "  # Predict result path
    cnt = print_directories(path_1)
    my_precision = 0.0
    my_recall = 0.0
    my_accuracy = 0.0
    my_f1 = 0.0

    for idx in range(1, cnt + 1):
        path_label = path_1 + "/label-" + str(idx) + ".tif"
        path_pre = path_2 + "/result-" + str(idx) + ".tif"
        diff = 10
        data_label = results2swc(path_label)
        data_pre = results2swc(path_pre)
        indexs = calTpFpFn(data_label, data_pre, diff)
        print("tp = ", indexs['tp'], ", fp = ", indexs['fp'], ", fn = ", indexs['fn'])
        metrics = claMetrics(indexs['tp'], indexs['fp'], indexs['fn'])
        print(metrics)
        my_precision += metrics['precision']/cnt
        my_recall += metrics['recall'] / cnt
        my_accuracy += metrics['accuracy'] / cnt
        my_f1 += metrics['f1'] / cnt
        print()

    print("")
    print("precision = ", my_precision)
    print("recall = ", my_recall)
    print("accuracy = ", my_accuracy)
    print("f1 = ", my_f1)



