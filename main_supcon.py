from __future__ import print_function

import os
import sys
import argparse
import time
import math
import pickle
import random
import copy

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model, label_convert
from dataUtil import osr_splits_inliers, get_train_datasets
from networks.resnet_big import SupConResNet
from networks.resnet_multi import SupConResNet_MultiHead
from networks.simCNN import simCNN_contrastive
from networks.resnet_preact import SupConpPreactResNet
from networks.mlp import SupConMLP
from losses import SupConLoss

import matplotlib
matplotlib.use('Agg')

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='1000',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet_multi',
                        choices=["resnet18", "resnet_multi", "resnet34", "preactresnet18", "preactresnet34", "simCNN", "MLP"])
    parser.add_argument("--last_model_path", type=str, default=None)
    parser.add_argument('--datasets', type=str, default='cifar10',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "tinyimgnet", 'mnist', "svhn", "cifar100_marco", "imagenet100"], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument("--augmentation_list", type=list, default=[])
    parser.add_argument("--argmentation_n", type=int, default=1)
    parser.add_argument("--argmentation_m", type=int, default=6)

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR', "SimCLR_CE", "MoCo"], help='choose method')
    parser.add_argument("--trail", type=int, default=0, choices=[0,1,2,3,4,5], help="index of repeating training")
    parser.add_argument("--action", type=str, default="training_supcon",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])
    # temperature
    parser.add_argument('--temp', type=float, default=0.05, help='temperature for loss')
    parser.add_argument('--temp1', type=float, default=0.005, help='temperature for loss')
    parser.add_argument('--temp2', type=float, default=0.01, help='temperature for loss')
    parser.add_argument('--temp3', type=float, default=0.05, help='temperature for loss')
    parser.add_argument("--out_dim", type=int, default=512, help="output dimension of the resnet blocks")
    parser.add_argument("--clip", type=float, default=None, help="for gradient clipping")

    # other setting
    parser.add_argument('--cosine', type=bool, default=False,
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument("--feat_dim", type=int, default=128)

    opt = parser.parse_args()

    opt.num_classes = len(osr_splits_inliers[opt.datasets][opt.trail])

    # check if dataset is path that passed required arguments
    if opt.datasets == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '../datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.datasets)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if "multi" in opt.model:
        opt.model_name = opt.datasets + "_" + opt.model + '_trail_{}'.format(opt.trail) + "_" + str(opt.feat_dim) + "_" + str(opt.out_dim) +  "_" + str(opt.temp1) + "_" + str(opt.temp2) +"_" + str(opt.temp3) + "_" + str(opt.batch_size)
    else:
        opt.model_name = opt.datasets + "_" + opt.model + '_trail_{}'.format(opt.trail) + "_" + str(opt.feat_dim) + "_" + str(opt.temp)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def set_loader(opt):
    # construct data loader

    train_dataset =  get_train_datasets(opt)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                                               num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    return train_loader


def set_model(opt):

    if opt.datasets == "mnist":
        in_channels = 1
    else:
        in_channels = 3

    if opt.model == "resnet18" or opt.model == "resnet34":
        model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    elif opt.model == "preactresnet18" or opt.model == "preactresnet34":
        model = SupConpPreactResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    elif opt.model == "MLP":
        model = SupConMLP(feat_dim=opt.feat_dim)
    elif opt.model == "resnet_multi":
        model = SupConResNet_MultiHead(output_dim=opt.out_dim, feat_dim=opt.feat_dim, in_channels=in_channels)
    else:
        model = simCNN_contrastive(opt, feature_dim=opt.feat_dim, in_channels=in_channels)

    if opt.last_model_path is not None:
        model = load_model(opt, model)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        cudnn.benchmark = True

    if opt.model == "resnet_multi":
        criterion1 = SupConLoss(temperature=opt.temp1)
        criterion2 = SupConLoss(temperature=opt.temp2)
        criterion3 = SupConLoss(temperature=opt.temp3)
        criterion1 = criterion1.cuda()
        criterion2 = criterion2.cuda()
        criterion3 = criterion3.cuda()
        return model, (criterion1, criterion2, criterion3)
    else:
        criterion = SupConLoss(temperature=opt.temp)
        criterion = criterion.cuda()
        return model, (criterion, None, None)


def load_model(opt, model=None):
    if model is None:
        model = SupConResNet(name=opt.model)

    ckpt = torch.load(opt.last_model_path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    return model



def train(train_loader, model, criterions, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()

    criterion1, criterion2, criterion3 = criterions

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):

        data_time.update(time.time() - end)
        images1 = images[0]
        images2 = images[1]
        
        images = torch.cat([images1, images2], dim=0)
        labels_convert = label_convert(labels, num_classes=opt.num_classes)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]

        # warm-up learning rate
        #warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        if opt.model == "resnet_multi":
            features1, features2 = model(images)
            features1_1, features1_2 = torch.split(features1, [bsz, bsz], dim=0)
            features2_1, features2_2 = torch.split(features2, [bsz, bsz], dim=0)
            #features3_1, features3_2 = torch.split(features3, [bsz, bsz], dim=0)
            features1 = torch.cat([features1_1.unsqueeze(1), features1_2.unsqueeze(1)], dim=1)
            features2 = torch.cat([features2_1.unsqueeze(1), features2_2.unsqueeze(1)], dim=1)
            #features3 = torch.cat([features3_1.unsqueeze(1), features3_2.unsqueeze(1)], dim=1)
            loss1 = criterion1(features1, labels)
            loss2 = criterion2(features2, labels)
            #loss3 = criterion3(features3, labels)
            loss = loss1 + loss2 #+ loss3
            losses1.update(loss1.item(), bsz)
            losses2.update(loss2.item(), bsz)
            #losses3.update(loss3.item(), bsz)
        else:
            features = model(images)
            features1, features2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
            loss = criterion1(features, labels)

        # update metric
        losses.update(loss.item(), bsz)
        # SGD
        optimizer.zero_grad()
        loss.backward()
        #plot_grad_flow(model.named_parameters(), idx, epoch)
        if opt.clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'loss1 {loss1.val:.3f} ({loss1.avg:.3f})\t'
                  'loss2 {loss2.val:.3f} ({loss2.avg:.3f})\t'
                   'loss3 {loss3.val:.3f} ({loss3.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, loss1=losses1, loss2=losses2, loss3=losses3))
            sys.stdout.flush()

    return losses.avg, losses1.avg, losses2.avg, losses3.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterions = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    losses = []
    losses1 = []
    losses2 = []
    losses3 = []

    # training routine
    for epoch in range(0, opt.epochs):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, loss1, loss2, loss3 = train(train_loader, model, criterions, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        losses.append(loss)
        losses1.append(loss1)
        losses2.append(loss2)
        losses3.append(loss3)
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
    with open(os.path.join(opt.save_folder, "loss_" + str(opt.trail)), "wb") as f:
         pickle.dump((losses, losses1, losses2, losses3), f)


if __name__ == '__main__':
    main()
