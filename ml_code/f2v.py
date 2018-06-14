import pickle
import sys
from csv import DictReader
import utils
import pefile
import capstone
from os.path import isfile
from collections import Counter


# load feature
ngrams_features_list = pickle.load(open('ml_code/features/ngrams_features'))
segments_features_set = pickle.load(open('ml_code/features/segments_features'))

UNK = '_UNK_'
BYTES_END = 'bytes'
ASM_END = 'asm'
DLL_END = 'dll'
SEGMENT_END = 'segments'

def produce_data_file_on_segments(dll_filename, dir_benign_dll):
    """
    Create a file that will provide information about the segments
        of the given file (name of segment and number of lines in it).
    The output file will have the same name as the input-file but instead, with '.segments' ending.

    :param dll_filename: name of dll file.
    """
    file_name = dll_filename.rsplit('/', 1)[1]
    try:
        pe = pefile.PE('%s/%s.%s' % (dir_benign_dll, file_name, DLL_END))
    except Exception as e:
        print 'Error with pefile on file: %s/%s.%s' % (dir_benign_dll, file_name, DLL_END)
        print e.message
        exit(0)
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    full_path = '%s/%s.%s' % ('ml_code/segments_data', file_name, SEGMENT_END)
    if not isfile(full_path):
        with open(full_path, 'w') as f:
            for section in pe.sections:
                code = section.get_data()
                first_instruction_address = section.PointerToRawData
                num_lines_in_section = 0
                for i in md.disasm(code, first_instruction_address):
                    num_lines_in_section += 1
                # for each section write in the file the name and number of lines in that section
                f.write('%s:%i\n' % (section.Name.strip('\0'), num_lines_in_section))


def count_seg_counts(f_name, seg_set, dir_benign_dll, dirpath='ml_code/segments_data'):
    """
    :param f_name: name of file.
    :param seg_set: set of segments-names.
    :return: dict that maps segment-name to number of lines in that segment in the given file.
    """
    name = f_name.rsplit('/', 1)[1]
    seg_counter = Counter()
    num_unks = 0  # number of unknown segments

    filepath_without_ending = '%s.' % (f_name)
    path_to_file = filepath_without_ending + ASM_END
    if isfile(path_to_file):  # can use .asm file
        mode = ASM_END
    else:  # has no .asm file, so parse the dll file and extract info about segments
        produce_data_file_on_segments(f_name, dir_benign_dll)
        path_to_file = dirpath + '/' + name + '.' + SEGMENT_END
        mode = SEGMENT_END

    with open(path_to_file, 'rb') as f:
        for line in f:
            seg_name, rest = line.split(':', 1)
            if seg_name not in seg_set:  # if it is unknown segment (was not in train set) mark it as UNK
                seg_name = UNK
                num_unks += 1

            if mode == ASM_END:  # in .asm file, the segment name appears for each line in it
                val_to_add = 1
            else:  # in .segments file, the segment name appears alongside the number of lines in it
                val_to_add = int(rest)
            seg_counter[seg_name] += val_to_add
    if num_unks > 0:  # for UNK segments, take the average number of lines
        seg_counter[UNK] = int(seg_counter[UNK] / num_unks)
    return seg_counter


def represent_file_as_vector(filename, dir_benign_dll):
    """
    :param filename: name of file, with the extension(= .bytes or .asm) .
    :return: vector of features that represents the given file.
    """
    vec = []

    # ngrams
    curr_ngrams_set = utils.get_ngrams_set_of(filename, n=4)
    for feature in ngrams_features_list:
        if feature in curr_ngrams_set:
            vec.append(1)
        else:
            vec.append(0)

    # segments
    seg_counter = count_seg_counts(filename, segments_features_set, dir_benign_dll)
    for seg_name in segments_features_set:
        if seg_name in seg_counter:
            vec.append(seg_counter[seg_name])
        else:
            vec.append(0)

    return vec


def create_file_file2vec(files_list, f2v_name, dir_benign_dll):
    """
    :param files_list: list of files-names.
    :param f2v_name: output file, will contain file-name and the vector that represents it.
        Format of file: filename<\t>vec
    """
    with open(f2v_name, 'w') as f:
        for f_name in files_list:
            vec = represent_file_as_vector(f_name, dir_benign_dll)  # represent each file as a vector
            vec = map(lambda x: str(x), vec)
            f.write(f_name + '\t' + ' '.join(vec) + '\n')


def main():
    """
    Parameters to main:
         f2l_filepath f2v_filepath dir_benign_dll
        # files_filepath - path to a .csv file, contains a column of 'Id' with file-name in each row.
        # f2v_filepath - path to f2v file. the name of the file to create.
                         will be in format of 'filename<tab>vector<EOL>'
        # dir_benign_dll - path to dir with dll files of benign files

    """
    args = sys.argv[1:]
    files_filepath = args[0]
    f2v_filepath = args[1]
    dir_benign_dll = args[2]

    # extract names of files
    file_list = []
    csv_dict = DictReader(open(files_filepath))
    for row in csv_dict:
        file_list.append(row['Id'])

    create_file_file2vec(file_list, f2v_filepath, dir_benign_dll)
    print 'done created %s' % f2v_filepath


if __name__ == '__main__':
    main()
