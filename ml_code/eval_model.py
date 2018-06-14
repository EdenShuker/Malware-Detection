from csv import DictReader
from time import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import sys

FILE_NAME = 'Id'
LABEL = 'Class'

SHOW_CONFUSION_MAT = '-show-matrix'

if __name__ == '__main__':
    """
    parameters to main:
        gold_filepath pred_filepath
        # gold_filepath - path to a .csv file with columns Id and Class
        # pred_filepath - same format as gold_filepath
    """

    t0 = time()

    args = sys.argv[1:]
    gold_filepath = args[0]
    pred_filepath = args[1]

    good = bad = 0
    gold_d = DictReader(open(gold_filepath))
    pred_d = DictReader(open(pred_filepath))

    gold_list, pred_list = [], []

    for gold, pred in zip(gold_d, pred_d):
        # check for same item comparison
        gold_name = gold[FILE_NAME]
        pred_name = pred[FILE_NAME]
        if gold_name != pred_name:
            print 'different names of files: %s %s' % (gold_name, pred_name)
            exit(0)

        gold_list.append(gold[LABEL])
        pred_list.append(pred[LABEL])

    # accuracy
    acc = accuracy_score(gold_list, pred_list)
    print 'accuracy: %0.2f%%' % (acc * 100.0)

    # confusion matrix
    if SHOW_CONFUSION_MAT in args:
        print confusion_matrix(gold_list, pred_list)

    print 'time to run evaluation:', time() - t0
