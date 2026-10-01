from __future__ import print_function

import os
import pickle
import sys
import argparse
import time
import math
import copy

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from util import AverageMeter
from main_supcon import set_loader
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, SupCEResNet, LinearClassifier
from networks.resnet_preact import SupConpPreactResNet
from networks.simCNN import simCNN_contrastive
from networks.mlp import SupConMLP
from dataUtil import osr_splits_inliers, get_train_datasets, get_test_datasets

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument("--action", type=str, default="trainging_linear",
                        choices=["training_supcon", "trainging_linear", 
                                 "testing_known", "testing_unknown", "feature_reading"])
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs')

    # optimization 
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18', choices=["resnet18", "resnet34", "vgg16", "simCNN", "MLP"])
    parser.add_argument('--datasets', type=str, default='tinyimgnet',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "cifar100", "tinyimgnet", 'mnist', "svhn", "cifar100_marco"], help='dataset')
    parser.add_argument("--backbone_model_direct", type=str, default="/save/SupCon/tinyimgnet_resnet18_trail_0_128_0.5/")
    parser.add_argument("--backbone_model_direct2", type=str, default=None)
    parser.add_argument("--backbone_model_direct3", type=str, default=None)
    parser.add_argument("--num_ensembles", type=int, default=1)
    parser.add_argument("--backbone_model_name", type=str, default="last.pth")
    parser.add_argument("--trail_backbone", type=int, default=0)
    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--temp_list", type=str, default="")

     # upsampling parameters
    parser.add_argument("--upsample", type=bool, default=False)
    parser.add_argument("--portion_out", type=float, default=0.5)
    parser.add_argument("--upsample_times", type=int, default=1)
    parser.add_argument("--last_feature_path", type=str, default=None)
    parser.add_argument("--last_model_path", type=str, default=None)
    parser.add_argument("--feat_dim", type=int, default=128)

    # other setting
    parser.add_argument('--cosine', type=bool, default=False,
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument("--seed", type=int, default=0)

    opt = parser.parse_args()
   
    opt.num_classes_backbone = len(osr_splits_inliers[opt.datasets][opt.trail_backbone])
    opt.num_classes = len(osr_splits_inliers[opt.datasets][opt.trail])

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.main_dir = os.getcwd()
    opt.backbone_model_direct = opt.main_dir + opt.backbone_model_direct
    opt.backbone_model_path = os.path.join(opt.backbone_model_direct, opt.backbone_model_name)
    if opt.backbone_model_direct2 is not None:
        opt.backbone_model_direct2 = opt.main_dir + opt.backbone_model_direct2
        opt.backbone_model_path2 = os.path.join(opt.backbone_model_direct2, opt.backbone_model_name)
    if opt.backbone_model_direct3 is not None:
        opt.backbone_model_direct3 = opt.main_dir + opt.backbone_model_direct3
        opt.backbone_model_path3 = os.path.join(opt.backbone_model_direct3, opt.backbone_model_name)

    opt.linear_model_path = os.path.join(opt.backbone_model_direct, "last_linear.pth")

    return opt


def load_model(model, path):
    ckpt = torch.load(path, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
    
    return model


def set_model(opt):
    criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(name=opt.model, num_classes=opt.num_classes, feat_dim=128*opt.num_ensembles)
    classifier = classifier.cuda()
    criterion = criterion.cuda()

    if opt.datasets == "mnist":
        in_channels = 1
    else:
        in_channels = 3

    if "cifar100_marco" in opt.datasets:
        model = SupCEResNet(name=opt.model, in_channels=in_channels, num_classes=opt.num_classes_backbone)
    else:
        if opt.model == "resnet18" or opt.model == "resnet34":
            model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
            if opt.backbone_model_direct2 is not None:
                model2 = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
            if opt.backbone_model_direct3 is not None:
                model3 = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
        elif opt.model == "preactresnet18" or opt.model == "preactresnet34":
            model = SupConpPreactResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
        elif opt.model == "MLP":
            model = SupConMLP(feat_dim=opt.feat_dim)
        else:
            model = simCNN_contrastive(opt, feature_dim=opt.feat_dim, in_channels=in_channels)

    model = load_model(model, opt.backbone_model_path)
    if opt.backbone_model_direct2 is not None and opt.backbone_model_direct3 is None:
        model2 = load_model(model2, opt.backbone_model_path2)
        return model, model2, None, classifier, criterion
    if opt.backbone_model_direct3 is not None:
        model2 = load_model(model2, opt.backbone_model_path2)
        model3 = load_model(model3, opt.backbone_model_path3)
        return model, model2, model3, classifier, criterion
    else:
        return model, None, None, classifier, criterion


def set_loader(opt):
    # construct data loader

    train_dataset =  get_train_datasets(opt)
    test_dataset = get_test_datasets(opt)
    train_dataset4test = get_train_datasets(opt)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                               num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                               num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)
    train_loader4test = torch.utils.data.DataLoader(train_dataset4test, batch_size=opt.batch_size, shuffle=False,
                                                    num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,
                                                    drop_last=True)
    return train_loader, test_loader, train_loader4test


def train(train_loader, model, model2, model3, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""

    model.eval()
    classifier.train()
    if model2 is not None:
        model2.eval()
    if model3 is not None:
        model3.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model(images)
            features = features.cuda(non_blocking=True)
            if model2 is not None:
                features2 = model2(images)
                features2 = features2.cuda(non_blocking=True)
                features = torch.cat((features, features2), dim=1)
            if model3 is not None:
                features3 = model3(images)
                features3 = features3.cuda(non_blocking=True)
                features = torch.cat((features, features3), dim=1)
        
        output = classifier(features)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, _ , _= accuracy(output, labels)
        top1.update(acc1, bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
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
                  'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, model2, model3, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()
    if model2 is not None:
        model2.eval()
    if model3 is not None:
        model3.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    preds = []

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            features = model(images)
            features = features.cuda(non_blocking=True)
            if model2 is not None:
                features2 = model2(images)
                features2 = features2.cuda(non_blocking=True)
                features = torch.cat((features, features2), dim=1)
            if model3 is not None:
                features3 = model3(images)
                features3 = features3.cuda(non_blocking=True)
                features = torch.cat((features, features3), dim=1)
            output = classifier(features)
            loss = criterion(output, labels)
            preds.append(torch.argmax(output.cpu(), dim=1).numpy())

            # update metric
            losses.update(loss.item(), bsz)
            acc1, _, _ = accuracy(output, labels)
            top1.update(acc1, bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    preds = np.array(preds)
    with open(os.path.join(opt.backbone_model_direct, "pred_out"), "wb") as f:
        pickle.dump(preds, f)
    return losses.avg, top1.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader, test_loader, train_loader4test = set_loader(opt)

    # build model and criterion
    model, model2, model3, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, model2, model3, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}, loss:{:.2f}'.format(
            epoch, time2 - time1, acc, loss))

    save_file = opt.backbone_model_name.replace(".pth", "_linear_") + opt.temp_list + ".pth"
    save_file = os.path.join(opt.backbone_model_direct, save_file)
    save_model(classifier, optimizer, opt, epoch, save_file)

    _, acc_val = validate(train_loader4test, model, model2, model3, classifier, criterion, opt)
    print('Evl accuracy:{:.2f}'.format(acc_val))


if __name__ == '__main__':
    main()
