import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import kornia

from torchvision import models

L1Loss = nn.L1Loss().cuda()

class VGGLoss(nn.Module):
    def __init__(self, gpu_id=0):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda(gpu_id)
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y, weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]):
        bs = x.size(0)
        while x.size()[3] > 1024:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, ignore_index=255,
                      weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def masked_distillation_loss(teacher_out, student_out, mask):
    teacher_out = F.softmax(teacher_out, dim=1)
    teacher_out = teacher_out.detach()
    loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=1), dim=1)  # map distillation
    return (loss * mask).sum() / (mask.sum() + 1e-6)


def distillation_loss(teacher_out, student_out, scale=0.25):
    student_out = student_out.chunk(2)
    teacher_out = F.softmax(teacher_out, dim=1)
    teacher_out = teacher_out.detach().chunk(2)
    total_loss = 0
    for iq, q in enumerate(teacher_out):
        for v in range(len(student_out)):
            if v == iq:
                # we skip cases where student and teacher operate on the same view
                continue
            # print('teacher_3:',q.size())
            # print('student_3:', student_out[v].size())
            # loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
            loss = torch.sum(-q * F.log_softmax(student_out[v], dim=1), dim=1)  # map distillation
            if iq == 1:
                loss*=scale
            #print('loss_shape', loss.size())
            total_loss += loss.mean()
    return total_loss



class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=255, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=True)

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):

        score = [score]

        weights = [1]
        assert len(weights) == len(score)

        functions = [self._ce_forward] * \
            (len(weights) - 1) + [self._ohem_forward]
        return sum([
            w * func(x, target)
            for (w, x, func) in zip(weights, score, functions)
        ])


def myL1Loss(source, target):
    return torch.mean(torch.abs(source - target))

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


def seg_edge_loss(rgb_out: torch.tensor, rgb_in: torch.tensor, segm_gt: torch.tensor):
    # rgb_gt    : batch_size x channels x width x height
    # rgb_pred  : batch_size x channels x width x height
    # segm_gt   : batch_size x 1 x width x height
    # w         : float ( <= 1)
    # check if the segmentation map is 3 or 4 dimensions
    if len(segm_gt.shape) == 3:
        segm_gt = segm_gt.unsqueeze(1)
    segm_gt = segm_gt.detach()

    # semantic edges
    segm_edges = kornia.laplacian(segm_gt.float(), kernel_size=5)
    segm_edges = (torch.abs(segm_edges) > 0.1).float()
    segm_edges = torch.cat((segm_edges, segm_edges, segm_edges), 1)  # ToDo: Need to expand if edges has only 1 channel

    # Sobel/Laplacian RGB edges
    edges = kornia.laplacian(rgb_in.float(), kernel_size=5)
    edges = (edges / edges.max() > 0.1).float()
    #edges = torch.cat((edges, edges, edges), 1)  # ToDo: Need to expand if edges has only 1 channel

    #print('segm_edges_shape', segm_edges.size() )
    #print('edges_shape', edges.size())
    #print('rgb_in_shape', rgb_in.size())
    #print('rgb_out_shape', rgb_out.size())

    assert segm_edges.size() == edges.size() == rgb_in.size() == rgb_out.size(), 'The shape of segm_edges, rgb edges, rgb in and rgb out are not same'

    loss = L1Loss(rgb_in * segm_edges, rgb_out * segm_edges)
    return loss, segm_edges, edges


