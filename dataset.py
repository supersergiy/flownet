import random
from copy import copy
import torch
import torch.utils.data as data

from PIL import Image

import os
import os.path
import sys
import io_routines

from log import logger


def split(all_ids_in, train_portion=0.85, val_portion=0.10, test_portion=0.05, seed=1):
    assert (train_portion + val_portion + test_portion == 1.0)

    logger.info("Generating a dataset split with random seed {}".format(seed))
    random.seed(seed)

    all_ids = copy(all_ids_in)
    n = len(all_ids)
    random.shuffle(all_ids)

    partition = {}
    train_start_id = 0
    train_stop_id  = int(n * train_portion)
    partition["train"] = all_ids[train_start_id:train_stop_id]

    val_start_id = int(n * train_portion)
    val_stop_id  = int(n * (train_portion + val_portion))
    partition["val"] = all_ids[val_start_id:val_stop_id]

    test_start_id = int(n * (train_portion + val_portion))
    test_stop_id  = n
    partition["test"] = all_ids[test_start_id:test_stop_id]

    return partition

def make_chairs_dataset(root, ids):
    samples = []

    for i in ids:
        name_1    = "{}_img1.ppm".format(i)
        name_2    = "{}_img2.ppm".format(i)
        name_flow = "{}_flow.flo".format(i)

        path_1 = os.path.join(root, name_1)
        path_2 = os.path.join(root, name_2)
        path_flow = os.path.join(root, name_flow)

        item = (path_1, path_2, path_flow)
        samples.append(item)

    return samples


def default_loader(path):
    return io_routines.read(path)

class FlyingChairsDataset(data.Dataset):
    """A flow data loader for flying chairs dataset::
        root/xxx.ext
        root/xxz.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        ids: the list of ids to use in this dataset
     Attributes:
        samples(list):
    """

    def __init__(self, root, ids, loader=default_loader,
                 transform=None, target_transform=None):

        samples = make_chairs_dataset(root, ids)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        self.root = root
        self.loader = loader

        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path1, path2, path_flow = self.samples[index]
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        flow = self.loader(path_flow)

        if self.transform is not None:
            img1 = self.transform(img1)
        if self.target_transform is not None:
            img2 = self.target_transform(img2)

        return img1, img2, flow

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



