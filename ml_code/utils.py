from csv import DictReader
from os import listdir
from os.path import isfile, join

UNK = '_UNK_'

BYTES_END = 'bytes'
ASM_END = 'asm'
DLL_END = 'dll'
SEGMENT_END = 'segments'


def get_ngrams_set_of(f_name, n=4):
    """
    :param f_name: file name.
    :param n: num of grams to concat.
    :return: set of ngrams of the given file.
    """
    path_to_file = "%s.%s" % (f_name, BYTES_END)
    one_list = []
    with open(path_to_file, 'rb') as f:
        for line in f:
            # append bytes to list
            line = line.rstrip().split(" ")
            line.pop(0)  # ignore address
            one_list += line

    # array holds all 4 grams opcodes (array of strings) . use sliding window.
    grams_list = [''.join(one_list[i:i + n]) for i in xrange(len(one_list) - n + 1)]

    # create a set of ngrams out of the ngrams
    ngrams_set = set()
    ngrams_set.update(grams_list)
    return ngrams_set


def get_files_from_dir(dirpath, ending):
    """
    :param dirpath: path to directory.
    :param ending: file-ending, the type of files you want to get.
    :return: list of files names that has the given ending.
    """
    end_len = len(ending)
    files = [f[:-end_len] for f in listdir(dirpath) if isfile(join(dirpath, f)) and f.endswith(ending)]
    return files



def read_csv(filepath, key1, key2):  # TODO documentation
    """
    :param filepath: path to f2l-file (train_labels_filtered.csv) .
    :param key1:
    :param key2:
    :return: file-to-label dict.
    """
    k2v_dict = dict()
    csv_dict = DictReader(open(filepath))
    for row in csv_dict:
        key = row[key1]
        val = row[key2]
        k2v_dict[key] = val
    return k2v_dict
