# coding: utf-8

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np

from PIL import Image

from torch.utils import data
from torchvision import transforms as T


class Dataset(data.Dataset):
  def __init__(self, root, data_list_file, phase='train', input_shape=(1, 128, 128)):
    self.phase = phase
    self.input_shape = input_shape

    with open(os.path.join(data_list_file), 'r') as fd:
      imgs = fd.readlines()

    self.imgs = np.random.permutation(imgs)

    if self.phase == 'train':
      self.transforms = T.Compose([
        T.RandomCrop(self.input_shape[1:]),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
      ])
    else:
      self.transforms = T.Compose([
        T.CenterCrop(self.input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
      ])

  def __getitem__(self, index):
    img_path, label = self.imgs[index].strip().split()
    data = self.transforms(Image.open(img_path).convert('L'))
    return data.float(), np.int32(label)

  def __len__(self):
    return len(self.imgs)


if __name__ == '__main__':
  dataset = Dataset(
    root=r'D:\dev\data\aligned-CASIA-WebFace',
    data_list_file=r'D:\dev\project\arcface-pytorch\train_list.txt',
    phase='train',
    input_shape=(1, 128, 128)
  )

  trainloader = data.DataLoader(dataset, batch_size=16)
  for i, (data, label) in enumerate(trainloader):
    print(data.shape, label)
