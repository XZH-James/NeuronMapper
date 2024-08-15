import os
import glob
import random
import shutil


def get_tif_files(directory):
    return glob.glob(os.path.join(directory, '*.tif'))


def split_dataset(file_list, train_ratio=0.67, val_ratio=0.16):
    random.shuffle(file_list)
    total = len(file_list)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_files = file_list[:train_end]
    val_files = file_list[train_end:val_end]
    test_files = file_list[val_end:]
    return train_files, val_files, test_files


def write_to_txt(file_list, filename):
    with open(filename, 'w') as f:
        for file in file_list:
            f.write(file + '\n')


def copy_files(file_list, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    copied_files = []
    for file in file_list:
        dest_file = os.path.join(destination_folder, os.path.basename(file))
        shutil.copy(file, dest_file)
        copied_files.append(dest_file)
    return copied_files


def main(directory, train_dir, val_dir, test_dir, train_file, val_file, test_file):
    random.seed()
    tif_files = get_tif_files(directory)
    train_files, val_files, test_files = split_dataset(tif_files)

    copied_train_files = copy_files(train_files, train_dir)
    copied_val_files = copy_files(val_files, val_dir)
    copied_test_files = copy_files(test_files, test_dir)

    write_to_txt(copied_train_files, train_file)
    write_to_txt(copied_val_files, val_file)
    write_to_txt(copied_test_files, test_file)


if __name__ == "__main__":
    directory = ''  # Replace with your TIF folder path
    train_dir = '../input/train'  # Replace with your input folder path
    val_dir = '../input/val'
    test_dir = '../input/test'
    train_file = '../input/train_path_list.txt'
    val_file = '../input/val_path_list.txt'
    test_file = '../input/test_path_list.txt'

    main(directory, train_dir, val_dir, test_dir, train_file, val_file, test_file)
