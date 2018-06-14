import utils
import pickle
from collections import Counter
from csv import DictReader
from sys import argv
from time import time

LABEL_NUM_FLAG = '-n'
TRAIN_LABELS_PATH_FLAG = '-p'
MOST_COMMON_NUM = 100000


def get_list_of_files_for_label(train_labels_path, label):
    """
    Extract from a file all the file-names with the given label.
    :param train_labels_path: path to trainLabels.csv file
    :param label: int, class number.
    :return: list of strings, each is a file-name which marked with the given label.
    """
    label = str(label)
    f_names = []
    for row in DictReader(open(train_labels_path)):
        if row['Class'] == label:
            f_names.append((row['Id']))
    return f_names


def get_most_common_ngrams(f_names):
    """
    :param f_names: list of file-names.
    :return: list of N most common ngrams out of all files.
    """
    ngrams_dict = Counter()  # count in how many files the ngrams appeared
    for f_name in f_names:
        curr_ngrams_set = utils.get_ngrams_set_of(f_name)  # ngrams set of current file
        for ngram in curr_ngrams_set:
            ngrams_dict[ngram] += 1
    return ngrams_dict.most_common(MOST_COMMON_NUM)


if __name__ == '__main__':
    """
    activate in the following order:
    python2.7 extract_ngrams.py path -n LABEL -p PATH
    for example:
    python2.7 extract_ngrams.py data/files -n 1 -p train_set.csv
    """
    t0 = time()

    # args from main
    label_num_index = argv.index(LABEL_NUM_FLAG) + 1
    label_num = int(argv[label_num_index])  # class-label to extract ngram for
    path_index = argv.index(TRAIN_LABELS_PATH_FLAG) + 1
    path = argv[path_index]  # path to trainLabels.csv that map file-name to its label

    file_names = get_list_of_files_for_label(path, label_num)
    # list of tuples, each tuple is (ngrams, num)
    most_common_ngrams = get_most_common_ngrams(file_names)
    pickle.dump(most_common_ngrams, open('ml_code/ngrams/label_%i' % label_num, 'wb'))

    print 'time to run label %i:' % label_num, time() - t0
