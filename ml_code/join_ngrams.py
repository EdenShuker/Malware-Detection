import heapq
import math
import pickle
from collections import Counter
from csv import DictReader

num_classes = 4


def count_ngrams_performances():
    """
    join all ngrams to one dictionary, which holds for every ngram
    the number of files it appears in for every class.
    :return: dict_all
    """
    ngram_to_num_occurs = dict()
    for c in range(num_classes):
        # de-serialize counter from file.
        grams_counter = pickle.load(open('ml_code/ngrams/label_%i' % c, 'rb'))  # list of tuples (ngram, num)
        grams_counter = dict(grams_counter)
        for gram in grams_counter:
            if gram not in ngram_to_num_occurs:
                # initialize an array: [0,0,0,0...]
                ngram_to_num_occurs[gram] = [0] * num_classes
            ngram_to_num_occurs[gram][c] = grams_counter[gram]
    return ngram_to_num_occurs


def num_instances(path):
    """
    :param path: path to 'trainLabels.csv'.
    :return: dict that map label-number to number of occurrences.
    """
    count_instances = Counter()
    for entry in DictReader(open(path)):
        count_instances[int(entry['Class'])] += 1
    return count_instances


def entropy(p, n):
    p_ratio = float(p) / (p + n)
    n_ratio = float(n) / (p + n)
    return -p_ratio * math.log(p_ratio) - n_ratio * math.log(n_ratio)


def compute_gain(p0, n0, p1, n1, p, n):
    return entropy(p, n) - float(p0 + n0) / (p + n) * entropy(p0, n0) - float(p1 + n1) / (p + n) * entropy(p1, n1)


def get_best_gain_features(p, n, class_label, dict_all, num_features=750, gain_minimum_bar=-100000):
    """
    extract the 750 best features from all 100,000
    :param p: number of files with the label- 'class_label'
    :param n: number of files with other labels.
    :param class_label: label
    :param dict_all: n_gram to number of performances.
    :param num_features:
    :param gain_minimum_bar:
    :return: 750 features with the biggest gain.
    """
    heap = [(gain_minimum_bar, 'gain_bar')] * num_features
    root = heap[0]
    for gram, count_list in dict_all.iteritems():
        p1 = count_list[class_label]
        # n1 - number of files with other labels with the n_gram.
        n1 = sum(count_list[:(class_label)] + count_list[class_label+1:])
        p0, n0 = p - p1, n - n1
        if p1 * p0 * n1 * n0 != 0:
            gain = compute_gain(p0, n0, p1, n1, p, n)
            if gain > root[0]:
                # pop smallest item and insert new one.
                root = heapq.heapreplace(heap, (gain, gram))
    return [i[1] for i in heap]


def get_selected_features():
    dict_all = count_ngrams_performances()
    counter_instances = num_instances('data/train_set.csv')
    num_all = sum(counter_instances.values())
    features_all = []
    for label in range(num_classes):
        p = counter_instances[label]
        n = num_all - p
        features_all += get_best_gain_features(p, n, label, dict_all)  # 750 * 9
    return features_all


if __name__ == '__main__':
    features = get_selected_features()
    pickle.dump(features, open('ml_code/features/ngrams_features', 'w'))
    print 'done joining ngrams'
