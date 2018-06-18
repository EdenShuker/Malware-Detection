import glob
from csv import DictReader
import torch
import torch.nn as nn
import numpy as np


def create_path2label_dict(dir_path, labels2dir_file):
    """
    :param dir_path: (string) path to directory that contains sub directories,
                     each contains files of their respective family.
    :param labels2dir_file: (string) path to a '.csv' file with the columns 'Label', 'Dir',
                            maps a label to the name of sub directory in dir_path.
    :return: (dict) maps from file-path (string) to its label.
    """
    path2label = dict()
    dir2label = DictReader(open(labels2dir_file))
    for row in dir2label:
        dir_name = row['Dir']
        label = row['Label']
        files = glob.glob(dir_path + '/' + dir_name + '/*.bytes')
        for f in files:
            path2label[f] = label
    return path2label


def read_lines(filepath):
    """
    :param filepath: (string) path to a file, each line will be an item.
    :return: (list).
    """
    ls = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            ls.append(line)
    return ls


def model_to_cuda(model):
    """
    :param model: (MalConv).
    :return: the device used for the model.
    """
    device = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda:0")
        model = nn.DataParallel(model)
        model.to(device)
    return device


def split_data_set(path2label):
    """
    :param path2label: (dict) maps file-path to its label.
    :return: 2 data sets of tuples (file_path, label) - train_set and dev_set.
    """
    keys = list(path2label.viewkeys())
    np.random.shuffle(keys)
    data_set = [(key, path2label[key]) for key in keys]
    split_ratio = int(np.floor(len(keys) * 0.8))
    train_set = data_set[:split_ratio]
    dev_set = data_set[split_ratio:]
    return train_set, dev_set


def split_to_files_and_labels(data_set):
    """
    :param: (list) data set of tuples, each is (file_path, label).
    :return: two lists, first is of the Id-column and second of Class-column.
    """
    fps = []
    labels = []
    for f_path, label in data_set:
        fps.append(f_path)
        labels.append(label)
    return fps, labels

