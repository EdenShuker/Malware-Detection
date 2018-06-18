from torch.utils.data import Dataset
from torch import tensor


class ExeDatasetNoLabels(Dataset):
    def __init__(self, fp_list, l2i, first_n_byte=2000000):
        """
        :param fp_list: list of strings, each is file-path.
        :param l2i: dict that maps label to index.
        :param first_n_byte: number of bytes to read from each file.
        """
        self.fp_list = fp_list
        self.l2i = l2i
        self.first_n_byte = first_n_byte

    def __len__(self):
        return len(self.fp_list)

    @staticmethod
    def represent_bytes(bytes_str):
        """
        :param bytes_str: string of bytes, i.e. two characters, each is one of 0-f.
        :return: integer, number between 0 to 257.
        """
        if bytes_str == '??':  # ignore those signs
            return 0
        return int(bytes_str, 16) + 1

    def __getitem__(self, idx):
        with open(self.fp_list[idx], 'r') as f:
            tmp = []
            for line in f:
                line = line.split()
                line.pop(0)  # ignore address

                line = map(ExeDataset.represent_bytes, line)
                tmp.extend(line)

            # padding with zeroes such that all files will be of the same size
            if len(tmp) > self.first_n_byte:
                tmp = tmp[:self.first_n_byte]
            else:
                tmp = tmp + [0] * (self.first_n_byte - len(tmp))
        f.close()
        return tensor(tmp)


class ExeDataset(ExeDatasetNoLabels):
    def __init__(self, fp_list, label_list, l2i, first_n_byte=2000000):
        """
        :param fp_list: list of strings, each is file-path.
        :param label_list: list of labels, each is integer.
        :param l2i: dict that maps label to index.
        :param first_n_byte: number of bytes to read from each file.
        """
        super(ExeDataset, self).__init__(fp_list, l2i, first_n_byte)
        self.label_list = label_list

    def __getitem__(self, idx):
        x = super(ExeDataset, self).__getitem__(idx)
        y = tensor(self.l2i[int(self.label_list[idx])])
        return x, y
