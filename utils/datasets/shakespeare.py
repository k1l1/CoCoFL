import numpy as np
import torchvision
import os
import json
from tqdm import tqdm
import torch
import multiprocessing
import copy
import itertools
import matplotlib.pyplot as plt

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
ALL_LETTERS += "ö"
NUM_LETTERS = len(ALL_LETTERS)


def _one_hot(index, size):
    """Returns one-hot vector with given size and value 1 at given index."""
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    """Returns one-hot representation of given letter."""
    index = max(0, ALL_LETTERS.find(letter))  # treating ' ' as unknown character
    return _one_hot(index, NUM_LETTERS)


def word_to_indices(word):
    '''returns a list of character indices
    Args:
        word: string
    
    Return:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(max(0, ALL_LETTERS.find(c)))  # added max to account for -1
    return indices


def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch


def process_y(raw_y_batch):
    y_batch = [word_to_indices(c) for c in raw_y_batch]  # to indices
    # y_batch = [letter_to_vec(c) for c in raw_y_batch]  # to one-hot
    return y_batch


def process(item):
            inputs = []
            for input_ in item[0]:
                input_ = process_x(input_)
                inputs.append(input_.reshape(-1))

            targets = []
            for target in item[1]:
                target = process_y(target)
                targets += target[0]
            return inputs, targets


class SHAKESPEARE(torchvision.datasets.VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(SHAKESPEARE, self).__init__(root, transform=transform,
                                          target_transform=target_transform)
        self.train = train

        self.train_data_dir = os.path.join(root, 'train')
        self.test_data_dir = os.path.join(root, 'test')

        self.classes = ALL_LETTERS

        data = {}

        suffix = "train" if self.train else "test"

        with open(root + "shakespeare/shakespeare_" + suffix + ".json", "r") as inf:
            cdata = json.load(inf)
        data.update(cdata["user_data"])

        list_keys = list(data.keys())
        self.inputs = []
        self.targets = []
        self.class_idxs = []

        raw = []
        for i in range(len(list_keys)):
            raw.append((data[list_keys[i]]["x"], data[list_keys[i]]["y"]))

        idx = 0
        with multiprocessing.Pool(processes=min(12, len(list_keys))) as pool:
            for result in pool.map(process, raw):
                self.inputs += result[0]
                self.targets += result[1]
                self.class_idxs.append(np.arange(idx, idx + len(result[1])))
                idx += len(result[1])

        self.targets = torch.tensor(np.array(self.targets))

    def __getitem__(self, index):
        input, target = self.inputs[index], int(self.targets[index])

        if self.target_transform is not None:
            target = self.target_transform(target)

        return torch.cat((torch.tensor(input), torch.atleast_1d(torch.tensor(80)))), target

    def __len__(self):
        return int(self.targets.shape[0])