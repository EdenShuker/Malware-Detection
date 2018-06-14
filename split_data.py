from collections import OrderedDict
from operator import itemgetter
import sys
import random
from ml_code import utils


def train_test_split(path_to_files, target_dir_path, test_size):    # todo documentation
    """
    split given labeled data to train set and test set.
    :param path_to_files: path to csv file which holds list of all files.
    :param target_dir_path: path to dir where the files will be created in.
    :param test_size: (float) the relative size of the test set from data.
    """
    files_to_labels = utils.read_csv(path_to_files, 'Id', 'Class')
    labels_to_files = get_labels_to_files(files_to_labels)
    train_set = {}
    test_set = {}
    for label in labels_to_files:
        files_with_label = labels_to_files[label]
        num_samples = len(files_with_label)

        if num_samples == 1:  # only one sample of that label, add it to train
            train_set[files_with_label[0]] = label
            continue

        # separate the indexes into train and test
        num_test_samples = int(num_samples * test_size)
        if num_test_samples == 0:
            num_test_samples = 1
        samples_indexes = range(0, num_samples)
        random.shuffle(samples_indexes)
        test_samples_indexes = samples_indexes[:num_test_samples]
        train_samples_indexes = samples_indexes[num_test_samples:]

        # add to each list the needed samples from X and the connected label
        for test_sample_i in test_samples_indexes:
            test_set[files_with_label[test_sample_i]] = label
        for train_sample_i in train_samples_indexes:
            train_set[files_with_label[train_sample_i]] = label

    save_to_csv_file(train_set, 'train_set', target_dir_path)
    save_to_csv_file(test_set, 'test_set', target_dir_path)


def save_to_csv_file(files_set, f_name, dir_path):
    """
    Save to csv file all files names and labels from files_set.
    :param files_set: dict from files to labels.
    :param f_name: csv file's name.
    :param dir_path: the path to the dir we want to create this files in it.
    """
    csv_f = open(dir_path + '/' + f_name + '.csv', 'w')
    csv_f.write('Id,Class\n')
    files_set = OrderedDict(sorted(files_set.items(), key=itemgetter(1)))
    for f in files_set.iterkeys():
        csv_f.write('%s,%s\n' % (f, files_set[f]))  # write the file and its label
    csv_f.close()


def get_labels_to_files(files_to_labels):
    """
    create reverse dict from labels to list of files.
    :param files_to_labels: dict from file to its label.
    :return: dict from label to list of files with that label.
    """
    labels_to_files = dict()
    for f in files_to_labels:
        label = files_to_labels[f]
        if label not in labels_to_files:
            labels_to_files[label] = [f]
        else:
            labels_to_files[label].append(f)
    return labels_to_files


if __name__ == '__main__':
    """
    args:    csv_file_path target_dir_path split_percentage l2dir_file_path
    example: data/train_labels_filtered.csv data 0.33 data/label2dir.csv
    """
    train_test_split(sys.argv[1], sys.argv[2], float(sys.argv[3]))
    print 'done splitting the data'
