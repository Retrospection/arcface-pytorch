# coding: utf-8

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import time
import numpy as np

import torch
import torch.nn as nn

from torch.nn import DataParallel
from torch.utils import batch_data
from torch.optim.lr_scheduler import StepLR, MultiStepLR


from batch_data.dataset import Dataset
from utils.visualizer import Visualizer
from config.config import Config
from test import lfw_test, get_lfw_list


from models.focal_loss import FocalLoss
from models.metrics import ArcMarginProduct, AddMarginProduct, SphereProduct
from models.resnet import resnet_face18, resnet34, resnet50



def save_model(model, save_path, name, iter_cnt):
  save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
  torch.save(model.state_dict(), save_name)
  return save_name


if __name__ == '__main__':
  config = Config()

  if config.display:
    visualizer = Visualizer()

  # ---------------------------------- prepare data ------------------------------

  train_dataset = Dataset(config.train_root, config.train_list, phase='train', input_shape=config.input_shape)
  train_loader = batch_data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers)

  test_identity_list = get_lfw_list(config.lfw_test_list)
  test_img_paths = [os.path.join(config.lfw_root, each) for each in test_identity_list]

  print('{} train iters per epoch:'.format(len(train_loader)))

  # ---------------------------------- prepare backbone network ------------------------------

  if config.backbone == 'resnet18':
    model = resnet_face18(use_se=config.use_se)
  elif config.backbone == 'resnet34':
    model = resnet34()
  elif config.backbone == 'resnet50':
    model = resnet50()

  # ---------------------------------- prepare classify loss ------------------------------

  if config.loss == 'focal_loss':
    criterion = FocalLoss(gamma=2)
  else:
    criterion = torch.nn.CrossEntropyLoss()

  # ---------------------------------- prepare metric loss ------------------------------

  if config.metric == 'add_margin':
    metric_fc = AddMarginProduct(512, config.num_classes, s=30, m=0.35)
  elif config.metric == 'arc_margin':
    metric_fc = ArcMarginProduct(512, config.num_classes, s=30, m=0.5, easy_margin=config.easy_margin)
  elif config.metric == 'sphere':
    metric_fc = SphereProduct(512, config.num_classes, m=4)
  else:
    metric_fc = nn.Linear(512, config.num_classes)

  # ---------------------------------- prepare optimizer ------------------------------

  if config.optimizer == 'sgd':
    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=config.lr,
                                momentum=0.9, weight_decay=config.weight_decay)
  else:
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=config.lr,
                                 weight_decay=config.weight_decay)

  # scheduler = StepLR(optimizer, step_size=conf.lr_step, gamma=0.1)
  scheduler = MultiStepLR(optimizer, milestones=[50, 70, 100], gamma=0.1)



  # ---------------------------------- prepare device ------------------------------

  device = torch.device("cuda")

  model.to(device)
  model = DataParallel(model)

  metric_fc.to(device)
  metric_fc = DataParallel(metric_fc)

  print(model)

# ---------------------------------- train ------------------------------

  start = time.time()
  for current_epoch in range(config.max_epoch):
    scheduler.step()

    model.train()
    for current_epoch_step, batch_data in enumerate(train_loader):

      train_image_batch, label = batch_data

      train_image_batch = train_image_batch.to(device)
      label = label.to(device).long()

      batch_feature = model(train_image_batch)
      predict_value = metric_fc(batch_feature, label)
      loss = criterion(predict_value, label)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      current_global_step = current_epoch * len(train_loader) + current_epoch_step

      if current_global_step % config.log_interval == 0:

        predict_value = predict_value.data.cpu().numpy()
        print(predict_value)
        predict_value = np.argmax(predict_value, axis=1)
        print(predict_value)
        label = label.data.cpu().numpy()
        # print(output)
        # print(label)
        acc = np.mean((predict_value == label).astype(int))
        speed = config.log_interval / (time.time() - start)

        time_str = time.asctime(time.localtime(time.time()))
        print('{} || train epoch {} || iter {} || {} iters/s || loss {} || acc {}'.format(time_str, current_epoch, current_epoch_step, speed, loss.item(), acc))

        if config.display:
          visualizer.display_current_results(current_global_step, loss.item(), name='train_loss')
          visualizer.display_current_results(current_global_step, acc, name='train_acc')

        start = time.time()

    if current_epoch % config.save_interval == 0 or current_epoch == config.max_epoch:
      save_model(model, config.checkpoints_path, config.backbone, current_epoch)

    model.eval()
    acc = lfw_test(model, test_img_paths, test_identity_list, config.lfw_test_list, config.test_batch_size)
    if config.display:
      visualizer.display_current_results(current_global_step, acc, name='test_acc')
