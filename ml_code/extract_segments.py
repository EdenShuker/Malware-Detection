import pickle
from time import time
import pefile
import utils
from sys import argv


def get_segment_set_of(dirpath, train_set_path, dir_benign_bytes):
    """
    :param dirpath: path to directory.
    :param train_set_path: path to trainLabels.csv .
    :return: set of segments-names extracted from all the files in the given directory.
    """
    seg_set = set()
    train_set = utils.read_csv(train_set_path, 'Id', 'Class').viewkeys()
    # segments from .asm files
    ASM_END = utils.ASM_END
    asm_files = utils.get_files_from_dir(dirpath, '.' + ASM_END)  # get list of .asm files
    for asm_f in asm_files:
        full_path = dirpath + '/' + asm_f
        if full_path in train_set:
            with open('%s.%s' % (full_path, ASM_END)) as f:
                for line in f:
                    segment_name = line.split(':', 1)[0]
                    seg_set.add(segment_name.rstrip('\x00'))

    # segments from .dll files
    DLL_END = utils.DLL_END
    dll_files = utils.get_files_from_dir(dirpath, '.' + DLL_END)  # get list of .dll files
    for dll_f in dll_files:
        full_path = dir_benign_bytes + '/' + dll_f
        if full_path in train_set:
            try:
                pe = pefile.PE('%s/%s.%s' % (dirpath, dll_f, DLL_END))
            except Exception as e:
                print 'Error with pefile on file: %s' % dll_f
                print e.message
                continue
            for section in pe.sections:
                seg_set.add(section.Name.rstrip('\x00'))
    return seg_set


if __name__ == '__main__':
    """
    activate in the following order:
    python2.7 extract_segments.py [path_to_malware files] [path_to_benign_dll] [path_to_benign_bytes]
    python ml_code/extract_segments.py /media/user/New Volume/train /media/user/New Volume/benign data/benign_bytes
    """
    t0 = time()

    dir_malware = argv[1]
    dir_benign_dll = argv[2]
    dir_benign_bytes = argv[3]
    segments1 = get_segment_set_of(dir_malware, 'data/train_set.csv', dir_benign_bytes)
    segments2 = get_segment_set_of(dir_benign_dll, 'data/train_set.csv', dir_benign_bytes)
    segments1.update(segments2)
    segments1.add(utils.UNK)
    print segments1
    pickle.dump(segments1, open('ml_code/features/segments_features', 'w'))

    print 'time to run:', time() - t0
