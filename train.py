# coding: utf-8

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from data.dataset import Dataset
from torch.utils import data
from utils.visualizer import Visualizer
from torch.optim.lr_scheduler import StepLR
from test import *
from models.metrics import *
from models.focal_loss import FocalLoss
from models.resnet import resnet_face18, resnet34, resnet50


def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


if __name__ == '__main__':

    conf = Config()

    print(conf)

    if conf.display:
        visualizer = Visualizer()

    device = torch.device("cuda")

    train_dataset = Dataset(conf.train_root, conf.train_list, phase='train', input_shape=conf.input_shape)
    trainloader = data.DataLoader(train_dataset, batch_size=conf.train_batch_size, shuffle=True, num_workers=conf.num_workers)

    identity_list = get_lfw_list(conf.lfw_test_list)
    img_paths = [os.path.join(conf.lfw_root, each) for each in identity_list]

    print('{} train iters per epoch:'.format(len(trainloader)))

    if conf.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if conf.backbone == 'resnet18':
        model = resnet_face18(use_se=conf.use_se)
    elif conf.backbone == 'resnet34':
        model = resnet34()
    elif conf.backbone == 'resnet50':
        model = resnet50()

    if conf.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, conf.num_classes, s=30, m=0.35)
    elif conf.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, conf.num_classes, s=30, m=0.5, easy_margin=conf.easy_margin)
    elif conf.metric == 'sphere':
        metric_fc = SphereProduct(512, conf.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, conf.num_classes)

    print(model)

    model.to(device)
    model = DataParallel(model)

    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if conf.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=conf.lr, weight_decay=conf.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=conf.lr, weight_decay=conf.weight_decay)
    scheduler = StepLR(optimizer, step_size=conf.lr_step, gamma=0.1)

    start = time.time()
    for i in range(conf.max_epoch):
        scheduler.step()

        model.train()
        for ii, data in enumerate(trainloader):

            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()

            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % conf.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                speed = conf.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))
                if conf.display:
                    visualizer.display_current_results(iters, loss.item(), name='train_loss')
                    visualizer.display_current_results(iters, acc, name='train_acc')

                start = time.time()

        if i % conf.save_interval == 0 or i == conf.max_epoch:
            save_model(model, conf.checkpoints_path, conf.backbone, i)

        model.eval()
        acc = lfw_test(model, img_paths, identity_list, conf.lfw_test_list, conf.test_batch_size)
        if conf.display:
            visualizer.display_current_results(iters, acc, name='test_acc')