import os
import torch
import h5py
import random
from PIL import Image
from torch.utils import data


class DataSet(data.Dataset):
    def __init__(self, h5_file_root):
        super(DataSet, self).__init__()
        self.h5_file = h5_file_root


    @staticmethod
    def random_horizontal_flip(lr, hr):
        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[2])
            hr = torch.flip(hr, dims=[2])
        #print("random_horizontal_fliprandom_horizontal_fliprandom_horizontal_flip")
        return lr, hr

    @staticmethod
    def random_vertical_flip(lr, hr):
        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[1])
            hr = torch.flip(hr, dims=[1])
        #print("random_vertical_fliprandom_vertical_fliprandom_vertical_flip")
        return lr, hr

    @staticmethod
    def random_rotation(lr, hr):
        if random.random() < 0.5:
            
            lr = torch.rot90(lr, dims=(2, 1))
            hr = torch.rot90(hr, dims=(2, 1))
        #print("random_rotationrandom_rotationrandom_rotation")
        return lr, hr

    def __getitem__(self, index):
        with h5py.File(self.h5_file, 'r') as f:
            hr = torch.from_numpy(f['hr'][str(index)][::])
            lr = torch.from_numpy(f['lr'][str(index)][::])
            #lr, hr = self.random_crop(lr, hr, self.patch_size, self.scale)
            lr, hr = self.random_vertical_flip(lr, hr)
            lr, hr = self.random_horizontal_flip(lr, hr)
            lr, hr = self.random_rotation(lr, hr)
            return lr, hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['hr'])


class ValidDataset(data.Dataset):
    def __init__(self, h5_file_root):
        super(ValidDataset, self).__init__()
        self.h5_root = h5_file_root

    def __getitem__(self, index):
        with h5py.File(self.h5_root, 'r') as f:
            hr = torch.from_numpy(f['hr'][str(index)][::])
            lr = torch.from_numpy(f['lr'][str(index)][::])

            return lr, hr

    def __len__(self):
        with h5py.File(self.h5_root, 'r') as f:
            return len(f['hr'])

