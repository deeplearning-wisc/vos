import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np

class RouteFcMaxAct(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, topk=5, conv1x1=False):
        super(RouteFcMaxAct, self).__init__(in_features, out_features, bias)
        if conv1x1:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features, 1, 1))
        self.topk = topk

    def forward(self, input):
        vote = input[:, None, :] * self.weight.squeeze()
        if self.bias is not None:
            out = vote.topk(self.topk, 2)[0].sum(2) + self.bias
        else:
            out = vote.topk(self.topk, 2)[0].sum(2)
        return out



class RouteFcUCPruned(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, topk=90, conv1x1=False, info=None):
        super(RouteFcUCPruned, self).__init__(in_features, out_features, bias)
        if conv1x1:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features, 1, 1))
        self.topk = topk
        self.info = info[0]
        self.masked_w = None

    def calculate_mask_weight(self):
        self.thresh = np.percentile(self.info, self.topk)
        mask = torch.Tensor((self.info > self.thresh))
        self.masked_w = (self.weight.squeeze().cpu() * mask).cuda()

    def forward(self, input):
        if self.masked_w is None:
            self.calculate_mask_weight()
        vote = input[:, None, :] * self.masked_w.cuda()
        if self.bias is not None:
            out = vote.sum(2) + self.bias
        else:
            out = vote.sum(2)
        return out


class RouteFcWtPruned(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, topk=90, conv1x1=False, info=None):
        super(RouteFcWtPruned, self).__init__(in_features, out_features, bias)
        if conv1x1:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features, 1, 1))
        self.topk = topk
        self.masked_w = None

    def calculate_mask_weight(self):
        w = self.weight.squeeze()
        thresh = np.percentile(w.data.cpu().numpy(), self.topk)
        mask = w > thresh
        self.masked_w = (self.weight.squeeze() * mask).cuda()

    def forward(self, input):
        if self.masked_w is None:
            self.calculate_mask_weight()
        vote = input[:, None, :] * self.masked_w
        if self.bias is not None:
            out = vote.sum(2) + self.bias
        else:
            out = vote.sum(2)
        return out


class RouteFcWtAbsPruned(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, topk=90, conv1x1=False, info=None):
        super(RouteFcWtAbsPruned, self).__init__(in_features, out_features, bias)
        if conv1x1:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features, 1, 1))
        self.topk = topk
        self.masked_w = None

    def calculate_mask_weight(self):
        w = self.weight.squeeze()
        thresh = np.percentile(w.abs().data.cpu().numpy(), self.topk)
        mask = w.abs() > thresh
        self.masked_w = (self.weight.squeeze() * mask).cuda()

    def forward(self, input):
        if self.masked_w is None:
            self.calculate_mask_weight()
        vote = input[:, None, :] * self.masked_w
        if self.bias is not None:
            out = vote.sum(2) + self.bias
        else:
            out = vote.sum(2)
        return out


class RouteUnitPruned(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, topk=5, info=None, conv1x1=False):
        super(RouteUnitPruned, self).__init__(in_features, out_features, bias)
        if conv1x1:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features, 1, 1))
        self.topk = topk
        self.info = info
        self.calculate_mask_unit()

    def calculate_mask_unit(self):
        unit_avg = self.info[0].mean(1)
        thresh = np.percentile(unit_avg, self.topk)
        self.mask = torch.Tensor(unit_avg > thresh).float().cuda()

    def forward(self, input):
        masked_input = input * self.mask
        vote = masked_input[:, None, :] * self.weight.squeeze()
        if self.bias is not None:
            out = vote.sum(2) + self.bias
        else:
            out = vote.sum(2)
        return out


class RouteUnitL1Pruned(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, topk=5, info=None, conv1x1=False):
        super(RouteUnitL1Pruned, self).__init__(in_features, out_features, bias)
        if conv1x1:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features, 1, 1))
        self.topk = topk
        self.info = info
        self.mask = None

    def calculate_mask_unit(self):
        w = self.weight.squeeze().data.cpu().numpy()
        # l1 = np.abs(w).mean(0)
        l2 = np.linalg.norm(w, 2, axis=0)
        thresh = np.percentile(l2, self.topk)
        self.mask = torch.Tensor(l2 > thresh).float().cuda()

    def forward(self, input):
        if self.mask is None:
            self.calculate_mask_unit()
        masked_input = input * self.mask
        vote = masked_input[:, None, :] * self.weight.squeeze()
        if self.bias is not None:
            out = vote.sum(2) + self.bias
        else:
            out = vote.sum(2)
        return out



class RouteTopkMax(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, topk=5, conv1x1=False):
        super(RouteTopkMax, self).__init__(in_features, out_features, bias)
        if conv1x1:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features, 1, 1))
        self.topk = topk

    def forward(self, input):
        vote = input[:, None, :] * self.weight.squeeze()
        fullsum = vote.sum(2)
        topksum = vote.topk(self.topk, 2)[0].sum(2)
        max_mask = fullsum == fullsum.max(1, keepdims=True)[0]
        if self.bias is not None:
            out = topksum * max_mask.float() + fullsum * (1 - max_mask.float()) + self.bias
        else:
            out = topksum * max_mask.float() + fullsum * (1 - max_mask.float())
        return out


class RouteDropout(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, p=50, conv1x1=False):
        super(RouteDropout, self).__init__(in_features, out_features, bias)
        if conv1x1:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features, 1, 1))
        self.p = p

    def forward(self, input):
        input = F.dropout(input, self.p / 100, training=True)
        return super(RouteDropout, self).forward(input)
        # vote = input[:, None, :] * self.weight.squeeze()
        # mask = torch.cuda.FloatTensor(vote.shape).uniform_() > self.p / 100
        # if self.bias is not None:
        #     out = (vote * mask).sum(2) + self.bias
        # else:
        #     out = (vote * mask).sum(2)
        # return out




class RouteFcWard(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, topk=90, conv1x1=False, info=None):
        super(RouteFcWard, self).__init__(in_features, out_features, bias)
        if conv1x1:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features, 1, 1))
        self.topk = topk
        self.info = info
        self.masked_w = None

    def calculate_mask_weight(self):
        mean, std = self.info
        ward = mean / (1e-1 + std)
        ward = np.abs(ward)
        self.thresh = np.percentile(ward, self.topk)
        mask = torch.Tensor((ward > self.thresh))
        self.masked_w = (self.weight.squeeze().cpu() * mask).cuda()

    def forward(self, input):
        if self.masked_w is None:
            self.calculate_mask_weight()
        vote = input[:, None, :] * self.masked_w.cuda()
        if self.bias is not None:
            out = vote.sum(2) + self.bias
        else:
            out = vote.sum(2)
        return out


#
# class RouteUnitAct(nn.Linear):
#
#     def __init__(self, in_features, out_features, bias=True, topk=5):
#         super(RouteUnitAct, self).__init__(in_features, out_features, bias)
#         self.topk = topk
#
#     def forward(self, input):
#         a = input[:, None, :] * 1/self.weight
#         val, _ = a.topk(self.topk + 1, 2)
#         mask = a > val[:, :, [-1]]
#         vote = input[:, None, :] * self.weight * mask
#         if self.bias is not None:
#             out = vote.sum(2) + self.bias
#         else:
#             out = vote.sum(2)
#         return out


# class RouteFcAbsMaxAct(nn.Linear):
#
#     def __init__(self, in_features, out_features, bias=True, topk=5, conv1x1=False):
#         super(RouteFcAbsMaxAct, self).__init__(in_features, out_features, bias)
#         if conv1x1:
#             self.weight = nn.Parameter(torch.Tensor(out_features, in_features, 1, 1))
#         self.topk = topk
#
#     def forward(self, input):
#         vote = input[:, None, :] * self.weight.squeeze()
#         val, ind = vote.abs().sort(2, descending=True)
#         ind_sel = ind[:, :, :self.topk]
#         vote_sel = torch.gather(vote, 2, ind_sel)
#         if self.bias is not None:
#             out = vote_sel.sum(2) + self.bias
#         else:
#             out = vote_sel.sum(2)
#         return out