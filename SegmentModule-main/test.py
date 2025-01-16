import csv
import os
import numpy as np
import SimpleITK as sitk
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects


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


def calculate_traditional_fpr(tp, fp, fn, total_pixels):
    """
    Traditional FPR
    FPR = FP / (FP + TN)
    """
    tn = total_pixels - tp - fp - fn
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def calculate_background_fpr(gt_path, pred_path):
    """
    Background-Based FPR
    FPR = False Positives / Total Background Pixels
    """
    if not os.path.exists(gt_path) or not os.path.exists(pred_path):
        print(f"Warning: {gt_path} or {pred_path} does not exist. Skipping...")
        return 0.0

    gt_image = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
    pred_image = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))

    background_mask = gt_image == 0
    total_background_pixels = np.sum(background_mask)

    false_positives = np.sum((pred_image > 0) & background_mask)

    return false_positives / total_background_pixels if total_background_pixels > 0 else 0.0


def results2swc(path):
    if not os.path.exists(path):
        print(f"Warning: {path} does not exist. Skipping...")
        return np.array([])
    seg = sitk.ReadImage(path)
    datanp_array = sitk.GetArrayFromImage(seg)
    datanp_array = label(datanp_array > 0.5)
    datanp_array = remove_small_objects(datanp_array, min_size=15)
    regions = regionprops(datanp_array)
    soma_loc_swc = []
    for rid, r in enumerate(regions):
        x, y, z = r.centroid
        soma_loc_swc.append([(rid + 1), 0, int(z), int(y), int(x), 0, -1])
    return np.array(soma_loc_swc)


def calTpFpFn(ori, pre, diff):
    tp = 0
    distances = []
    cnt_ori = ori.shape[0]
    cnt_pre = pre.shape[0]

    for vec1 in ori:
        for vec2 in pre:
            distance = np.linalg.norm(vec1 - vec2)
            if distance <= diff:
                tp += 1
                distances.append(distance)
                break

    avg_distance = np.mean(distances) if distances else 0.0
    return {'tp': tp, 'fp': cnt_ori - tp, 'fn': cnt_pre - tp, 'avg_distance': avg_distance}


def claMetrics(tp, fp, fn):
    return {
        'precision': precision(tp, fp, fn),
        'recall': recall(tp, fp, fn),
        'accuracy': accuracy(tp, fp, fn),
        'f1': f1(tp, fp, fn)
    }


if __name__ == "__main__":
    path_1 = "../dataset/test/label/"  # Replace with the GT label directory
    path_2 = "../experiments/result"  # Replace with the predict results directory
    save_path = "../experiments/model"  # Result save directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # CSV 文件路径
    csv_file = os.path.join(save_path, "evaluation_results.csv")

    cnt = print_directories(path_1)
    diff = 10

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(["Index", "Precision", "Recall", "Accuracy", "F1", "Traditional FPR", "Background FPR", "Average Distance"])

        my_precision = 0.0
        my_recall = 0.0
        my_accuracy = 0.0
        my_f1 = 0.0
        my_traditional_fpr = 0.0
        my_background_fpr = 0.0
        my_avg_distance = 0.0

        for idx in range(1, cnt + 1):
            path_label = os.path.join(path_1, f"label-{idx}.tif")
            path_pre = os.path.join(path_2, f"result-{idx}.tif")
            data_label = results2swc(path_label)
            data_pre = results2swc(path_pre)
            indexs = calTpFpFn(data_label, data_pre, diff)
            metrics = claMetrics(indexs['tp'], indexs['fp'], indexs['fn'])

            total_pixels = 512 * 512 * 512  # Replace with the actual total number of pixels
            traditional_fpr = calculate_traditional_fpr(indexs['tp'], indexs['fp'], indexs['fn'], total_pixels)
            background_fpr = calculate_background_fpr(path_label, path_pre)

            my_precision += metrics['precision'] / cnt
            my_recall += metrics['recall'] / cnt
            my_accuracy += metrics['accuracy'] / cnt
            my_f1 += metrics['f1'] / cnt
            my_traditional_fpr += traditional_fpr / cnt
            my_background_fpr += background_fpr / cnt
            my_avg_distance += indexs['avg_distance'] / cnt

            writer.writerow([idx, metrics['precision'], metrics['recall'], metrics['accuracy'], metrics['f1'], traditional_fpr, background_fpr, indexs['avg_distance']])

        writer.writerow(["Average", my_precision, my_recall, my_accuracy, my_f1, my_traditional_fpr, my_background_fpr, my_avg_distance])

    print(f"Results saved to {csv_file}")
