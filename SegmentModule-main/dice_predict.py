import os
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
import csv
from sklearn.metrics import jaccard_score


class DiceAverage(object):
    """Computes and stores the average and current value for calculating average Dice"""

    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0] * self.class_num, dtype='float64')
        self.avg = np.asarray([0] * self.class_num, dtype='float64')
        self.sum = np.asarray([0] * self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = self.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        for class_index in range(targets.size(1)):  # Assuming logits and targets are in (B, C, D, H, W) format
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dices.append(dice.item())
        return np.asarray(dices)


def to_one_hot_3d(tensor, n_classes=2):  # shape = [s, h, w] for a single sample
    # Ensure tensor is compatible with PyTorch
    if tensor.dtype != np.uint8:
        tensor = tensor.astype(np.uint8)  # Convert to uint8

    tensor = torch.from_numpy(tensor)  # Convert numpy array to tensor
    s, h, w = tensor.shape
    one_hot = torch.zeros(n_classes, s, h, w)  # One-hot encoding
    for class_idx in range(n_classes):
        one_hot[class_idx] = (tensor == class_idx).float()  # Convert boolean to float
    return one_hot


def calculate_dice(gt_path, pred_path, n_labels):
    gt_img = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
    pred_img = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))

    if np.sum(gt_img) == 0 and np.sum(pred_img) == 0:
        print(f"Warning: Both GT and prediction are empty for {gt_path} and {pred_path}.")
        return np.array([1.0] * n_labels)

    # Convert to one-hot encoding
    gt_one_hot = to_one_hot_3d(gt_img, n_labels)
    pred_one_hot = to_one_hot_3d(pred_img, n_labels)

    # Ensure the tensor is a PyTorch tensor
    gt_tensor = gt_one_hot.unsqueeze(0)
    pred_tensor = pred_one_hot.unsqueeze(0)

    # Initialize DiceAverage object
    dice_calculator = DiceAverage(n_labels)
    dice_calculator.update(pred_tensor, gt_tensor)

    return dice_calculator.avg


def evaluate_dice_for_directory(gt_dir, pred_dir, n_labels, output_csv_path):
    """
    Evaluate Dice scores for all pairs of GT and predicted images in the given directories.
    Also, save the Dice scores to a CSV file.
    """
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.tif')])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.tif')])

    assert len(gt_files) == len(pred_files), "GT and Prediction directories must contain the same number of files."

    dice_scores = {}

    # Prepare the CSV file for writing the results
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Filename', 'Dice_Score_Class_0', 'Dice_Score_Class_1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for gt_file, pred_file in zip(gt_files, pred_files):
            gt_path = os.path.join(gt_dir, gt_file)
            pred_path = os.path.join(pred_dir, pred_file)

            # Calculate Dice score for the current file pair
            dice = calculate_dice(gt_path, pred_path, n_labels)
            dice_scores[gt_file] = dice

            print(f"Dice scores for {gt_file}: {dice}")

            # Write the result to CSV
            writer.writerow({
                'Filename': gt_file,
                'Dice_Score_Class_0': dice[0],  # Dice score for class 0
                'Dice_Score_Class_1': dice[1],  # Dice score for class 1
            })

    return dice_scores


if __name__ == '__main__':
    # Specify the directories containing GT and predicted TIFF images
    gt_dir = '../dataset/test/label/'  # Replace with the GT label directory
    pred_dir = "../experiments/result"  # Replace with the predict results directory
    n_labels = 2  # Specify the number of classes (labels)
    output_csv_path = "../experiments/model"  # Result save directory

    dice_scores = evaluate_dice_for_directory(gt_dir, pred_dir, n_labels, output_csv_path)

    # Print out the overall Dice scores
    print("\nDice scores for all images:")
    for filename, dice in dice_scores.items():
        print(f"{filename}: {dice}")
