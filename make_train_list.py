# coding: utf-8

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os

DATASET_ROOT = r'/home/cuda80/yrs/train_data/webface-align-182-png'

with open('train_list.txt', 'w') as f:
  fullpaths = []
  for root, personIds, _ in os.walk(DATASET_ROOT):
    for index, personId in enumerate(personIds):
      for personRoot, _, images in os.walk(os.path.join(DATASET_ROOT, personId)):
        fullpaths.extend(['{} {}\n'.format(os.path.join(DATASET_ROOT, personId, image), index) for image in images])
  f.writelines(fullpaths)
