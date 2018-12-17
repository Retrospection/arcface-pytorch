# coding: utf-8

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


class Config(object):

  env = 'default'
  backbone = 'resnet18'
  classify = 'softmax'
  num_classes = 10575
  metric = 'arc_margin'
  easy_margin = True
  use_se = True
  loss = 'focal_loss'

  display = False
  finetune = False

  train_root = r'/home/cuda80/yrs/train_data/webface-align-182-png'
  train_list = r'/home/cuda80/yrs/20181211/arcface-pytorch/train_list.txt'
  val_list = r'/home/cuda80/yrs/20181211/arcface-pytorch/train_list.txt'

  test_root = r'/home/cuda80/yrs/train_data/webface-align-182-png'
  test_list = r'/home/cuda80/yrs/20181211/arcface-pytorch/train_list.txt'

  lfw_root = r'/home/cuda80/yrs/test_data/lfw_home/lfw-align-128'
  lfw_test_list = r'/home/cuda80/yrs/20181211/arcface-pytorch/lfw_test_pair.txt'

  checkpoints_path = 'checkpoints'
  # load_model_path = './checkpoints/20181214_201050_sgd/resnet18_20.pth'
  load_model_path = ''
  # test_model_path = './checkpoints/pretrained/resnet18_110.pth'
  test_model_path = ''
  save_interval = 10

  train_batch_size = 256  # batch size
  test_batch_size = 64

  input_shape = (1, 128, 128)

  optimizer = 'sgd'

  use_gpu = True  # use GPU or not
  gpu_id = [0, 1]
  num_workers = 8  # how many workers for loading data
  log_interval = 100  # print info every N batch

  debug_file = './debug'  # if os.path.exists(debug_file): enter ipdb
  result_file = 'result.csv'

  max_epoch = 300
  lr = 0.1  # initial learning rate
  lr_step = 10
  lr_decay = 0.98  # when val_loss increase, lr = lr*lr_decay
  weight_decay = 5e-4
