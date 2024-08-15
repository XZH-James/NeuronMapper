"""
This code is used to train the network to complete the task of classifying whether an input image contains somata.
After the training is complete, the code tests the classification effect on three different datasets (training/validation/testing).
"""


import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from model.models import NMClassify
from model.dataset import somata
from model.evaluator import getAUC, getACC, save_results
import pandas as pd


def main(flag, input_root, output_root, end_epoch, download):
    """
    main function
    :param flag: name of subset

    """

    dataclass = {
        "somata": somata
    }

    n_channels = 1
    n_classes = 2
    task = 'binary-class'
    start_epoch = 0
    lr = 0.001
    batch_size = 2
    val_auc_list = []
    dir_path = os.path.join(output_root, '%s_checkpoints' % (flag))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

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
    train_dataset = dataclass[flag](root=input_root, split='train', transform=train_transform, download=download)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = dataclass[flag](root=input_root, split='val', transform=val_transform, download=download)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = dataclass[flag](root=input_root, split='test', transform=test_transform, download=download)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    print('==> Building and training model...')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NMClassify(in_channels=n_channels, num_classes=n_classes).to(device)
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(start_epoch, end_epoch):
        train(model, optimizer, criterion, train_loader, device, task)
        val(model, val_loader, device, val_auc_list, task, dir_path, epoch)

    auc_list = np.array(val_auc_list)
    index = auc_list.argmax()
    print('epoch %s is the best model' % (index))

    print('==> Testing model...')
    restore_model_path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
    model.load_state_dict(torch.load(restore_model_path)['net'])
    test(model, 'train', train_loader, device, flag, task, output_root=output_root)
    test(model, 'val', val_loader, device, flag, task, output_root=output_root)
    test(model, 'test', test_loader, device, flag, task, output_root=output_root)


def train(model, optimizer, criterion, train_loader, device, task):
    """
    training function
    :param model: the model to train
    :param optimizer: optimizer used in training
    :param criterion: loss function
    :param train_loader: DataLoader of training set
    :param device: cpu or cuda
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class
    """

    model.train()
    for batch_idx, (inputs, targets, fileName) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long().to(device)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


def val(model, val_loader, device, val_auc_list, task, dir_path, epoch):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    file_name = np.array([])
    with torch.no_grad():
        for batch_idx, (inputs, targets, fileName) in enumerate(val_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                if targets.dim() == 0:
                    targets = targets.unsqueeze(0)
                targets = targets.float().view(-1, 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

            # Make sure the fileName length is sufficient
            if len(fileName) > 1:
                file_name = np.append(file_name, fileName[0])
                file_name = np.append(file_name, fileName[1])
            else:
                for f in fileName:
                    file_name = np.append(file_name, f)

        y_true = y_true.cpu().numpy()
        y_score_2 = y_score.cpu().numpy()
        file_name_pd = pd.DataFrame(file_name)
        y_score_pd = pd.DataFrame(y_score_2)
        result = pd.concat([file_name_pd, y_score_pd], axis=1)
        # Saves the path to verify the classification effect of the set during training
        pd.DataFrame(result).to_csv(
            "../output/val-file-name.csv")
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        val_auc_list.append(auc)

    state = {
        'net': model.state_dict(),
        'auc': auc,
        'epoch': epoch,
    }

    path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch, auc))
    torch.save(state, path)


def test(model, split, data_loader, device, flag, task, output_root=None):
    model.eval()
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
                if targets.dim() == 0:
                    targets = targets.unsqueeze(0)
                targets = targets.float().view(-1, 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

            # Make sure the fileName length is sufficient
            if len(fileName) > 1:
                file_name = np.append(file_name, fileName[0])
                file_name = np.append(file_name, fileName[1])
            else:
                for f in fileName:
                    file_name = np.append(file_name, f)

        y_true = y_true.cpu().numpy()
        y_score_2 = y_score.cpu().numpy()
        file_name_pd = pd.DataFrame(file_name)
        y_score_pd = pd.DataFrame(y_score_2)
        result = pd.concat([file_name_pd, y_score_pd], axis=1)

        # Save the path to the dataset classification effect during the test
        pd.DataFrame(result).to_csv(
            "../output/" + split + "-file-name.csv")
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        acc = getACC(y_true, y_score, task)
        print('%s AUC: %.5f ACC: %.5f' % (split, auc, acc))

        if output_root is not None:
            output_dir = os.path.join(output_root, flag)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, '%s.csv' % (split))
            save_results(y_true, y_score, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RUN Baseline model of MedMNIST')
    parser.add_argument('--data_name', default='somata', help='subset of MedMNIST', type=str)

    # This is the input path
    parser.add_argument('--input_root', default='/../input',
                        help='input root, the source of dataset files', type=str)

    # This is the output path
    parser.add_argument('--output_root', default='./output/NMC', help='output root, where to save models and results',
                        type=str)

    # The number of epochs trained is set here
    parser.add_argument('--num_epoch', default=100, help='num of epochs of training', type=int)

    parser.add_argument('--download', default=False, help='whether download the dataset or not', type=bool)

    args = parser.parse_args()
    data_name = args.data_name.lower()
    input_root = args.input_root
    output_root = args.output_root
    end_epoch = args.num_epoch
    download = args.download
    main(data_name, input_root, output_root, end_epoch=end_epoch, download=download)
