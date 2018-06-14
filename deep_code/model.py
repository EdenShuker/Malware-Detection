from csv import DictReader
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import glob
import random

from utils import ExeDataset

log = open('log.txt', 'w')
log.write('train-accuracy,train-loss,dev-accuracy\n')
num_classes = 4


class MalConv(nn.Module):
    def __init__(self, labels, input_length=2000000, window_size=500):
        """
        :param labels: list of optional labels.
        :param input_length: length of input.
        :param window_size: size of window
        """
        super(MalConv, self).__init__()

        self.embed = nn.Embedding(257, 8, padding_idx=0)

        self.conv_1 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)

        self.pooling = nn.MaxPool1d(int(input_length / window_size))

        self.fc_1 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.3)

        self.sigmoid = nn.Sigmoid()
        # TODO: unnecessary when classes labels starts with 0.
        self.i2l = {i: l for i, l in enumerate(labels)}
        self.l2i = {l: i for i, l in self.i2l.iteritems()}

    def forward(self, x):
        """
        :param x: Variable input.
        :return: Variable output.
        """
        x = self.embed(x)
        # Channel first
        x = torch.transpose(x, -1, -2)

        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))

        x = self.relu(cnn_value * gating_weight)
        x = self.pooling(x)
        x = self.dropout(x)

        x = x.view(-1, 128)
        x = self.relu(self.fc_1(x))
        x = self.fc_2(x)

        return x


def split_to_files_and_labels(data_set):
    """
    :param: data set of tuples - (file_path, label)
    :return: two lists, first is of the Id-column and second of Class-column.
    """
    fps = []
    labels = []
    for f_path, label in data_set:
        fps.append(f_path)
        labels.append(label)
    return fps, labels


def split_data_set(path2label):
    """
    :param path2label: dictionary of file paths to their labels.
    :return: 2 data sets of tuples (file_path, label) - train_set and dev_set
    """
    keys = list(path2label.viewkeys())
    random.shuffle(keys)
    data_set = [(key, path2label[key]) for key in keys]
    split_ratio = int(np.floor(len(keys) * 0.8))
    train_set = data_set[:split_ratio]
    dev_set = data_set[split_ratio:]
    return train_set, dev_set


def train_on(path2label, first_n_byte=2000000, lr=0.001, verbose=True, num_epochs=3):
    """
    :param first_n_byte: number of bytes to read from each file.
    :param lr: learning rate.
    :param verbose: boolean.
    :param num_epochs: number of epochs.
    """
    # create model
    model = MalConv(range(1, num_classes + 1))
    device = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda:0")
        model = nn.DataParallel(model)
        model.to(device)

    # l2i = model.l2i
    # TODO: delete
    classes = range(1, num_classes + 1)
    i2l = {i: l for i, l in enumerate(classes)}
    l2i = {l: i for i, l in i2l.iteritems()}


    # load data
    train_set, dev_set = split_data_set(path2label)
    fps_train, y_train = split_to_files_and_labels(train_set)
    fps_dev, y_dev = split_to_files_and_labels(dev_set)

    # transfer data to DataLoader object
    dataloader = DataLoader(ExeDataset(fps_train, y_train, l2i, first_n_byte),
                            batch_size=32, shuffle=True, num_workers=10)
    validloader = DataLoader(ExeDataset(fps_dev, y_dev, l2i, first_n_byte),
                             batch_size=32, shuffle=False, num_workers=10)

    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    adam_optim = torch.optim.Adam(model.parameters(), lr)

    total_loss = 0.0
    valid_best_acc = 0.0
    total_step = 0
    test_step = 2500

    for epoch in range(num_epochs):
        t0 = time()
        good = 0.0
        model.train()

        for batch_data in dataloader:
            adam_optim.zero_grad()

            exe_input, label = batch_data[0], batch_data[1]
            if device is not None:
                exe_input, label = exe_input.to(device), label.to(device)
            pred = model(exe_input)

            loss = criterion(pred, label)
            total_loss += loss
            loss.backward()
            adam_optim.step()

            gold_label = label.data
            pred_label = torch.max(pred, 1)[1].data
            for y1, y2 in zip(gold_label, pred_label):
                if y1 == y2:
                    good += 1
                if verbose:
                    print 'gold: ', y1, ', pred: ', y2

            total_step += 1

            # if total_step % test_step == test_step - 1:  # interrupt for validation
            #     curr_acc = validate_dev_set(validloader, model, verbose)
            #     if curr_acc > valid_best_acc:  # update best accuracy
            #         valid_best_acc = curr_acc
            #         torch.save(model, 'model.file')
        acc_train = good / len(y_train)
        avg_loss_train = total_loss / len(y_train)
        print('{} TRN\ttime: {:.2f} accuracy: {}'.format(epoch, time() - t0, acc_train))
        acc_dev = validate_dev_set(validloader, model, device, len(y_dev), verbose)
        log.write('{:.4f},{:.5f},{:.4f}\n'.format(acc_train, avg_loss_train, acc_dev))
        # if acc_dev >= valid_best_acc:
        #     valid_best_acc = acc_dev
        #     torch.save(model, 'model.file')

    # conf matrix
    validate_dev_set(validloader, model, device, len(y_dev), verbose, conf=True)
    torch.save(model, 'model_final.file')


def validate_dev_set(valid_loader, model, device,size_dev, verbose=True, conf=False):
    """
    check performance of model on dev-set.
    :param valid_loader: DataLoader.
    :param model: Module which is the model to check with.
    :param verbose: boolean.
    :return: model accuracy on dev-set.
    """
    t0 = time()
    good = 0.0
    total_loss = 0.0
    model.eval()

    if conf:
        golds, preds = [], []

    for val_batch_data in valid_loader:
        exe_input, labels = val_batch_data[0], val_batch_data[1]
        if device is not None:
            exe_input, labels = exe_input.to(device), labels.to(device)

        pred = model(exe_input)

        gold_label = labels.data
        pred_label = torch.max(pred, 1)[1].data
        if conf:
            golds.extend(gold_label)
            preds.extend(pred_label)
        for y1, y2 in zip(gold_label, pred_label):
            if y1 == y2:
                good += 1
            if verbose:
                print 'gold: ', y1, ', pred: ', y2

    acc = good / size_dev
    print ' DEV\ttime:', time() - t0, ', accuracy:', acc * 100, '%\n'
    if conf:
        from sklearn.metrics import confusion_matrix
        print confusion_matrix(golds, preds)
    return acc


def get_data(dir_path):
    path2label = dict()
    dir2label = DictReader(open('../data/label2dir.csv'))  # todo change path
    for row in dir2label:
        dir_name = row['Dir']
        label = row['Label']
        files = glob.glob(dir_path + '/' + dir_name + '/*.bytes')
        for file in files:
            path2label[file] = label
    return path2label


if __name__ == '__main__':
    mode = 'train'
    path2label = get_data('/home/user/Desktop/Malware_data')

    if mode == 'train':
        train_on(path2label, verbose=False)

