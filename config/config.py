# coding: utf-8

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 13938
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False

    train_root = r'D:\dev\data\aligned-CASIA-WebFace'
    train_list = r'D:\dev\project\arcface-pytorch\train_list.txt'
    val_list = r'D:\dev\project\arcface-pytorch\train_list.txt'

    test_root = r'D:\dev\data\aligned-CASIA-WebFace'
    test_list = r'D:\dev\project\arcface-pytorch\train_list.txt'

    lfw_root = r'C:\Users\FOOTS\scikit_learn_data\lfw_home\lfw_funneled'
    lfw_test_list = r'D:\dev\project\arcface-pytorch\lfw_test_pair.txt'

    checkpoints_path = 'checkpoints'
    load_model_path = './models/resnet18.pth'
    test_model_path = './checkpoints/resnet18_40.pth'
    save_interval = 10

    train_batch_size = 96  # batch size
    test_batch_size = 64

    input_shape = (1, 128, 128)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0'
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = './debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 50
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
