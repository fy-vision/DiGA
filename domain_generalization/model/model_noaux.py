import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models

from torch.autograd import Variable
from .model_util import *
from .seg_model_noaux import DeeplabMulti

pspnet_specs = {
    'n_classes': 19,
    'input_size': (713, 713),
    'block_config': [3, 4, 23, 3],
}


class SegModel(nn.Module):
    def __init__(self, initialization=None, bn_clr=False):
        super(SegModel, self).__init__()
        self.n_classes = pspnet_specs['n_classes']
        self.bn_clr = bn_clr
        self.initialization = initialization

        model_seg = DeeplabMulti(pretrained=True, num_classes=self.n_classes, initialization = self.initialization, bn_clr =self.bn_clr)

        self.layer0 = nn.Sequential(model_seg.conv1, model_seg.bn1, model_seg.relu, model_seg.maxpool)
        self.layer1 = model_seg.layer1
        self.layer2 = model_seg.layer2
        self.layer3 = model_seg.layer3
        self.layer4 = model_seg.layer4
        if self.bn_clr:
            self.bn_pretrain = model_seg.bn_pretrain
        self.final = model_seg.layer5

    def forward(self, x):
        #inp_shape = x.shape[2:]

        x = self.layer0(x)
        # [2, 64, 65, 129]
        x = self.layer1(x)
        x = self.layer2(x)
        shared_shallow = x

        x = self.layer3(x)
        shared_seg = self.layer4(x)
        if self.bn_clr:
            shared_seg = self.bn_pretrain(shared_seg)
        out_dict = self.final(shared_seg)
        #print(out_dict)
        pred2 = out_dict['out']
        feat = out_dict['feat']

        return shared_shallow, shared_seg, pred2, feat

    def get_1x_lr_params_NOscale(self):
        b = []

        b.append(self.layer0)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        b = []
        if self.bn_clr:
            b.append(self.bn_pretrain.parameters())
        b.append(self.final.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': 1 * learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]
