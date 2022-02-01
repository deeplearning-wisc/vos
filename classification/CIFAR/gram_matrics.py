from __future__ import division,print_function

#matplotlib inline
#load_ext autoreload
#autoreload 2

import sys
from tqdm import tqdm_notebook as tqdm

import random
import matplotlib.pyplot as plt
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable, grad
from torchvision import datasets, transforms
from torch.nn.parameter import Parameter

import calculate_log as callog

import warnings
warnings.filterwarnings('ignore')


torch.cuda.set_device(0) #Select the GPU


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        t = self.conv1(x)
        out = F.relu(self.bn1(t))
        torch_model.record(t)
        torch_model.record(out)
        t = self.conv2(out)
        out = self.bn2(self.conv2(out))
        torch_model.record(t)
        torch_model.record(out)
        t = self.shortcut(x)
        out += t
        torch_model.record(t)
        out = F.relu(out)
        torch_model.record(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.collecting = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y

    def record(self, t):
        if self.collecting:
            self.gram_feats.append(t)

    def gram_feature_list(self, x):
        self.collecting = True
        self.gram_feats = []
        self.forward(x)
        self.collecting = False
        temp = self.gram_feats
        self.gram_feats = []
        return temp

    def load(self, path="resnet_cifar10.pth"):
        tm = torch.load(path, map_location="cpu")
        self.load_state_dict(tm)

    def get_min_max(self, data, power):
        mins = []
        maxs = []

        for i in range(0, len(data), 128):
            batch = data[i:i + 128].cuda()
            feat_list = self.gram_feature_list(batch)

            for L, feat_L in enumerate(feat_list):#96, x, x, x
                if L == len(mins):
                    mins.append([None] * len(power))
                    maxs.append([None] * len(power))

                for p, P in enumerate(power):
                    g_p = G_p(feat_L, P)

                    current_min = g_p.min(dim=0, keepdim=True)[0]
                    breakpoint()
                    current_max = g_p.max(dim=0, keepdim=True)[0]

                    if mins[L][p] is None:
                        mins[L][p] = current_min
                        maxs[L][p] = current_max
                    else:
                        mins[L][p] = torch.min(current_min, mins[L][p])
                        maxs[L][p] = torch.max(current_max, maxs[L][p])
        # breakpoint()
        return mins, maxs

    def get_deviations(self, data, power, mins, maxs):
        deviations = []

        for i in range(0, len(data), 128):
            batch = data[i:i + 128].cuda()
            feat_list = self.gram_feature_list(batch)

            batch_deviations = []
            for L, feat_L in enumerate(feat_list):
                dev = 0
                for p, P in enumerate(power):
                    g_p = G_p(feat_L, P)

                    dev += (F.relu(mins[L][p] - g_p) / torch.abs(mins[L][p] + 10 ** -6)).sum(dim=1, keepdim=True)
                    dev += (F.relu(g_p - maxs[L][p]) / torch.abs(maxs[L][p] + 10 ** -6)).sum(dim=1, keepdim=True)
                batch_deviations.append(dev.cpu().detach().numpy())
            batch_deviations = np.concatenate(batch_deviations, axis=1)
            deviations.append(batch_deviations)
        deviations = np.concatenate(deviations, axis=0)

        return deviations


torch_model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=10)
torch_model.load('/nobackup-slow/dataset/my_xfdu/resnet_cifar10.pth')
torch_model.cuda()
torch_model.params = list(torch_model.parameters())
torch_model.eval()
print("Done")

batch_size = 128
mean = np.array([[0.4914, 0.4822, 0.4465]]).T

std = np.array([[0.2023, 0.1994, 0.2010]]).T
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize

])
transform_test = transforms.Compose([
    transforms.CenterCrop(size=(32, 32)),
    transforms.ToTensor(),
    normalize
])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('/nobackup-slow/dataset/cifarpy', train=True, download=True,
                     transform=transform_train),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('/nobackup-slow/dataset/cifarpy', train=False, transform=transform_test),
    batch_size=batch_size)

data_train = list(torch.utils.data.DataLoader(
        datasets.CIFAR10('/nobackup-slow/dataset/cifarpy', train=True, download=True,
                       transform=transform_test),
        batch_size=1, shuffle=False))

data = list(torch.utils.data.DataLoader(
    datasets.CIFAR10('/nobackup-slow/dataset/cifarpy', train=False, download=True,
                   transform=transform_test),
    batch_size=1, shuffle=False))


torch_model.eval()
# correct = 0
# total = 0
# for x,y in test_loader:
#     x = x.cuda()
#     y = y.numpy()
#     correct += (y==np.argmax(torch_model(x).detach().cpu().numpy(),axis=1)).sum()
#     total += y.shape[0]
# print("Accuracy: ",correct/total)


cifar100 = list(torch.utils.data.DataLoader(
    datasets.CIFAR100('/nobackup-slow/dataset/cifarpy', train=False, download=True,
                   transform=transform_test),
    batch_size=1, shuffle=True))

train_preds = []
train_confs = []
train_logits = []
for idx in range(0, len(data_train), 128):
    batch = torch.squeeze(torch.stack([x[0] for x in data_train[idx:idx + 128]]), dim=1).cuda()

    logits = torch_model(batch)
    confs = F.softmax(logits, dim=1).cpu().detach().numpy()
    preds = np.argmax(confs, axis=1)
    logits = (logits.cpu().detach().numpy())

    train_confs.extend(np.max(confs, axis=1))
    train_preds.extend(preds)
    train_logits.extend(logits)
print("Done")

test_preds = []
test_confs = []
test_logits = []

for idx in range(0, len(data), 128):
    batch = torch.squeeze(torch.stack([x[0] for x in data[idx:idx + 128]]), dim=1).cuda()

    logits = torch_model(batch)
    confs = F.softmax(logits, dim=1).cpu().detach().numpy()
    preds = np.argmax(confs, axis=1)
    logits = (logits.cpu().detach().numpy())

    test_confs.extend(np.max(confs, axis=1))
    test_preds.extend(preds)
    test_logits.extend(logits)
print("Done")

import calculate_log as callog


def detect(all_test_deviations, all_ood_deviations, verbose=True, normalize=True):
    average_results = {}
    for i in range(1, 11):
        random.seed(i)

        validation_indices = random.sample(range(len(all_test_deviations)), int(0.1 * len(all_test_deviations)))
        test_indices = sorted(list(set(range(len(all_test_deviations))) - set(validation_indices)))

        validation = all_test_deviations[validation_indices]
        test_deviations = all_test_deviations[test_indices]

        t95 = validation.mean(axis=0) + 10 ** -7
        if not normalize:
            t95 = np.ones_like(t95)
        test_deviations = (test_deviations / t95[np.newaxis, :]).sum(axis=1)
        ood_deviations = (all_ood_deviations / t95[np.newaxis, :]).sum(axis=1)

        results = callog.compute_metric(-test_deviations, -ood_deviations)
        for m in results:
            average_results[m] = average_results.get(m, 0) + results[m]

    for m in average_results:
        average_results[m] /= i
    if verbose:
        callog.print_results(average_results)
    return average_results


def cpu(ob):
    for i in range(len(ob)):
        for j in range(len(ob[i])):
            ob[i][j] = ob[i][j].cpu()
    return ob


def cuda(ob):
    for i in range(len(ob)):
        for j in range(len(ob[i])):
            ob[i][j] = ob[i][j].cuda()
    return ob


class Detector:
    def __init__(self):
        self.all_test_deviations = None
        self.mins = {}
        self.maxs = {}

        self.classes = range(10)

    def compute_minmaxs(self, data_train, POWERS=[10]):
        for PRED in tqdm(self.classes):

            train_indices = np.where(np.array(train_preds) == PRED)[0]
            train_PRED = torch.squeeze(torch.stack([data_train[i][0] for i in train_indices]), dim=1)
            mins, maxs = torch_model.get_min_max(train_PRED, power=POWERS)
            self.mins[PRED] = cpu(mins)
            self.maxs[PRED] = cpu(maxs)
            torch.cuda.empty_cache()

    def compute_test_deviations(self, POWERS=[10]):
        all_test_deviations = None
        test_classes = []
        for PRED in tqdm(self.classes):
            test_indices = np.where(np.array(test_preds) == PRED)[0]
            test_PRED = torch.squeeze(torch.stack([data[i][0] for i in test_indices]), dim=1)
            test_confs_PRED = np.array([test_confs[i] for i in test_indices])

            test_classes.extend([PRED] * len(test_indices))

            mins = cuda(self.mins[PRED])
            maxs = cuda(self.maxs[PRED])

            test_deviations = torch_model.get_deviations(test_PRED, power=POWERS, mins=mins, maxs=maxs) / test_confs_PRED[:, np.newaxis]
            cpu(mins)
            cpu(maxs)
            if all_test_deviations is None:
                all_test_deviations = test_deviations
            else:
                all_test_deviations = np.concatenate([all_test_deviations, test_deviations], axis=0)
            torch.cuda.empty_cache()
        self.all_test_deviations = all_test_deviations

        self.test_classes = np.array(test_classes)

    def compute_ood_deviations(self, ood, POWERS=[10]):
        ood_preds = []
        ood_confs = []

        for idx in range(0, len(ood), 128):
            batch = torch.squeeze(torch.stack([x[0] for x in ood[idx:idx + 128]]), dim=1).cuda()
            logits = torch_model(batch)
            confs = F.softmax(logits, dim=1).cpu().detach().numpy()
            preds = np.argmax(confs, axis=1)

            ood_confs.extend(np.max(confs, axis=1))
            ood_preds.extend(preds)
            torch.cuda.empty_cache()
        print("Done")

        ood_classes = []
        all_ood_deviations = None
        for PRED in tqdm(self.classes):
            ood_indices = np.where(np.array(ood_preds) == PRED)[0]
            if len(ood_indices) == 0:
                continue
            ood_classes.extend([PRED] * len(ood_indices))

            ood_PRED = torch.squeeze(torch.stack([ood[i][0] for i in ood_indices]), dim=1)
            ood_confs_PRED = np.array([ood_confs[i] for i in ood_indices])
            mins = cuda(self.mins[PRED])
            maxs = cuda(self.maxs[PRED])
            ood_deviations = torch_model.get_deviations(ood_PRED, power=POWERS, mins=mins, maxs=maxs) / ood_confs_PRED[
                                                                                                        :, np.newaxis]
            cpu(self.mins[PRED])
            cpu(self.maxs[PRED])
            if all_ood_deviations is None:
                all_ood_deviations = ood_deviations
            else:
                all_ood_deviations = np.concatenate([all_ood_deviations, ood_deviations], axis=0)
            torch.cuda.empty_cache()

        self.ood_classes = np.array(ood_classes)
        breakpoint()
        average_results = detect(self.all_test_deviations, all_ood_deviations)
        return average_results, self.all_test_deviations, all_ood_deviations


def G_p(ob, p):
    temp = ob.detach()

    temp = temp ** p
    temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
    temp = ((torch.matmul(temp, temp.transpose(dim0=2, dim1=1)))).sum(dim=2)
    temp = (temp.sign() * torch.abs(temp) ** (1 / p)).reshape(temp.shape[0], -1)

    return temp


detector = Detector()
detector.compute_minmaxs(data_train, POWERS=range(1, 11))

detector.compute_test_deviations(POWERS=range(1, 11))

print("CIFAR-100")
c100_results = detector.compute_ood_deviations(cifar100,POWERS=range(1,11))