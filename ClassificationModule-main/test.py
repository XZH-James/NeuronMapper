"""
This code is used to test the network model trained by train.py.
Get the indicators used in the paper (AUC/ACC/network parameters/FPS)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import time

from model.models import NMClassify
from model.dataset import somata
from model.evaluator import getAUC, getACC, save_results
import pandas as pd


def main(flag, input_root, output_root, download):
    ''' main function
    :param flag: name of subset
    '''

    dataclass = {
        "somata": somata
    }

    n_channels = 1
    n_classes = 2
    task = 'binary-class'
    batch_size = 8

    print('==> Preparing data...')
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    print('==> transform...')
    test_dataset = dataclass[flag](root=input_root, split='test', transform=test_transform, download=download)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    print('==> Loading model...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NMClassify(in_channels=n_channels, num_classes=n_classes).to(device)

    # Modify this path to the correct model checkpoint path.
    # The first sets the model path, and the second sets the model name
    restore_model_path = os.path.join(
        '../output/NMC', '.pth')
    model.load_state_dict(torch.load(restore_model_path)['net'])
    model = torch.nn.DataParallel(model)

    print('==> Testing model...')
    test(model, 'test', test_loader, device, flag, task, output_root=output_root)


def test(model, split, data_loader, device, flag, task, output_root=None):
    """
    testing function
    :param model: the model to test
    :param split: the data to test, 'train/val/test'
    :param data_loader: DataLoader of data
    :param device: cpu or cuda
    :param flag: subset name
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class
    """

    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    file_name = np.array([])
    with torch.no_grad():
        for batch_idx, (inputs, targets, fileName) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)
            file_name = np.append(file_name, fileName)

        # 将数据转换为 NumPy 格式
        y_true = y_true.cpu().numpy()
        y_score_np = y_score.cpu().numpy()
        y_pred = np.argmax(y_score_np, axis=1)  # 获取每张图片的预测类别

        # 计算混淆矩阵元素
        tp = np.sum((y_pred == 1) & (y_true.flatten() == 1))
        tn = np.sum((y_pred == 0) & (y_true.flatten() == 0))
        fp = np.sum((y_pred == 1) & (y_true.flatten() == 0))
        fn = np.sum((y_pred == 0) & (y_true.flatten() == 1))

        # 计算 Sensitivity 和 Specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # 保存结果到 CSV
        file_name_pd = pd.DataFrame(file_name, columns=["FileName"])
        y_score_pd = pd.DataFrame(y_score_np, columns=[f"Score_{i}" for i in range(y_score_np.shape[1])])
        y_pred_pd = pd.DataFrame(y_pred, columns=['Predicted_Label'])
        result = pd.concat([file_name_pd, y_score_pd, y_pred_pd], axis=1)

        output_csv_path = f"../output/{split}-results.csv"
        result.to_csv(output_csv_path, index=False)

        print(f"保存{split}数据集的预测结果到 {output_csv_path}")

        # 计算 AUC 和 ACC
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        acc = getACC(y_true, y_score, task)
        print('%s AUC: %.5f ACC: %.5f Sensitivity: %.5f Specificity: %.5f' % (split, auc, acc, sensitivity, specificity))

        # 保存最终结果到特定路径
        if output_root is not None:
            output_dir = os.path.join(output_root, flag)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, f'{split}.csv')
            save_results(y_true, y_score, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RUN Baseline model of MedMNIST')
    parser.add_argument('--data_name', default='somata', help='subset of MedMNIST', type=str)

    # This is the input path
    parser.add_argument('--input_root', default='../input',
                        help='input root, the source of dataset files', type=str)

    # This is the output path
    parser.add_argument('--output_root', default='../output', help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--download', default=False, help='whether download the dataset or not', type=bool)

    args = parser.parse_args()
    data_name = args.data_name.lower()
    input_root = args.input_root
    output_root = args.output_root
    download = args.download
    main(data_name, input_root, output_root, download=download)
