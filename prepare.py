import PIL.Image
from PIL import Image
from torchvision import transforms as T
import torch
import os
import numpy as np
from tool import denormalize, convert_rgb_to_y, calculate_psnr
from config import opt
import cv2
import h5py


def create_h5_file(root, scale):
    h5_file = h5py.File('The path of trainout2. h5', 'w')
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
        
    img_names = os.listdir(root)
    img_paths = [os.path.join(root, name) for name in img_names]
    index = 0

    for img_path in img_paths:
        print(img_path)
        hr = Image.open(img_path).convert('RGB')
        lr = hr.resize((hr.width//scale, hr.height//scale), resample=PIL.Image.BICUBIC)
        hr = T.ToTensor()(hr)
        lr = T.ToTensor()(lr)
        lr_group.create_dataset(str(index), data=lr)
        hr_group.create_dataset(str(index), data=hr)
        index += 1

    h5_file.close()


def create_h5_file_valid(root, scale):
    h5_file = h5py.File('The path to evalout2. h', 'w')
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    valid_img_names = os.listdir(root)
    paths = [os.path.join(root, name) for name in valid_img_names]
    pos = 0

    for path in paths:
        hr = Image.open(path).convert('RGB')
        lr = hr.resize((hr.width//scale, hr.height//scale), resample=PIL.Image.BICUBIC)
        hr = T.ToTensor()(hr)
        lr = T.ToTensor()(lr)
        lr_group.create_dataset(str(pos), data=lr)
        hr_group.create_dataset(str(pos), data=hr)
        pos += 1
#         for hr in T.FiveCrop(size=(hr.height//2, hr.width//2))(hr):
#             hr = hr.resize(((hr.width//scale)*scale, (hr.height//scale)*scale), resample=PIL.Image.BICUBIC)
#             lr = hr.resize((hr.width//scale, hr.height//scale), resample=PIL.Image.BICUBIC)

#             hr = T.ToTensor()(hr)
#             lr = T.ToTensor()(lr)

            # lr_group.create_dataset(str(pos), data=lr)
            # hr_group.create_dataset(str(pos), data=hr)
            # pos += 1

    h5_file.close()
    
create_h5_file("Path to train dataset",2)
create_h5_file_valid("Path to valid dataset",2)