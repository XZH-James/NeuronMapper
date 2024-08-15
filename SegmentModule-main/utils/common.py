import torch
import random


def to_one_hot_3d(tensor, n_classes=2):
    """
    Convert a 3D tensor to one-hot encoding.
    Args:
        tensor (torch.Tensor): Input tensor with shape [batch, s, h, w].
        n_classes (int): Number of classes for one-hot encoding.
    Returns:
        torch.Tensor: One-hot encoded tensor with shape [batch, n_classes, s, h, w].
    """
    n, s, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, s, h, w, device=tensor.device)
    return one_hot.scatter_(1, tensor.view(n, 1, s, h, w), 1)


def random_crop_3d(img, label, crop_size):
    """
    Perform a random crop on a 3D image and its corresponding label.
    Args:
        img (numpy.ndarray): 3D image array.
        label (numpy.ndarray): Corresponding 3D label array.
        crop_size (tuple): Desired crop size (depth, height, width).
    Returns:
        tuple: Cropped image and label arrays, or None if crop is not possible.
    """
    d, h, w = img.shape
    crop_d, crop_h, crop_w = crop_size

    if any([d < crop_d, h < crop_h, w < crop_w]):
        return None

    max_d = d - crop_d
    max_h = h - crop_h
    max_w = w - crop_w

    start_d = random.randint(0, max_d)
    start_h = random.randint(0, max_h)
    start_w = random.randint(0, max_w)

    crop_img = img[start_d:start_d + crop_d, start_h:start_h + crop_h, start_w:start_w + crop_w]
    crop_label = label[start_d:start_d + crop_d, start_h:start_h + crop_h, start_w:start_w + crop_w]

    return crop_img, crop_label


def center_crop_3d(img, label, slice_num=16):
    """
    Perform a center crop on a 3D image and its corresponding label.
    Args:
        img (numpy.ndarray): 3D image array.
        label (numpy.ndarray): Corresponding 3D label array.
        slice_num (int): Number of slices to crop along the depth axis.
    Returns:
        tuple: Cropped image and label arrays, or None if crop is not possible.
    """
    d = img.shape[0]
    if d < slice_num:
        return None

    center = d // 2
    half_slice = slice_num // 2
    crop_img = img[center - half_slice:center + half_slice]
    crop_label = label[center - half_slice:center + half_slice]

    return crop_img, crop_label


def load_file_name_list(file_path):
    """
    Load a list of filenames from a file.
    Args:
        file_path (str): Path to the file containing filenames.
    Returns:
        list: List of filenames.
    """
    with open(file_path, 'r') as file_to_read:
        return [line.strip() for line in file_to_read if line.strip()]


def print_network(net):
    """
    Print the architecture and number of parameters in a network.
    Args:
        net (torch.nn.Module): Neural network model.
    """
    num_params = sum(p.numel() for p in net.parameters())
    print(net)
    print(f'Total number of parameters: {num_params}')


def adjust_learning_rate(optimizer, epoch, args):
    """
    Adjust the learning rate based on the epoch.
    Args:
        optimizer (torch.optim.Optimizer): Optimizer used in training.
        epoch (int): Current epoch number.
        args (argparse.Namespace): Arguments containing learning rate settings.
    """
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_fixed(optimizer, lr):
    """
    Set the learning rate to a fixed value.
    Args:
        optimizer (torch.optim.Optimizer): Optimizer used in training.
        lr (float): Learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
