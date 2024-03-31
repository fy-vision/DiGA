import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models

from torch.autograd import Variable
from .model_util import *
from .seg_model import DeeplabMulti

pspnet_specs = {
    'n_classes': 19,
    'input_size': (713, 713),
    'block_config': [3, 4, 23, 3],
}
'''
Sequential blocks
'''


class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()
        self.n_classes = pspnet_specs['n_classes']

        model_seg = DeeplabMulti(num_classes=self.n_classes)

        self.layer0 = nn.Sequential(model_seg.conv1, model_seg.bn1, model_seg.relu, model_seg.maxpool)
        self.layer1 = model_seg.layer1
        self.layer2 = model_seg.layer2
        self.layer3 = model_seg.layer3
        self.layer4 = model_seg.layer4

        self.final1 = model_seg.layer5
        self.final2 = model_seg.layer6

    def forward(self, x):
        #inp_shape = x.shape[2:]

        x = self.layer0(x)
        # [2, 64, 65, 129]
        x = self.layer1(x)
        x = self.layer2(x)
        shared_shallow = x

        x = self.layer3(x)
        pred1 = self.final1(x)

        shared_seg = self.layer4(x)
        out_dict = self.final2(shared_seg)
        #print(out_dict)
        pred2 = out_dict['out']
        feat = out_dict['feat']


        return shared_shallow, pred1, pred2, feat

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
        b.append(self.final1.parameters())
        b.append(self.final2.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': 1 * learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]


class Classifier(nn.Module):
    def __init__(self, inp_shape):
        super(Classifier, self).__init__()
        n_classes = pspnet_specs['n_classes']
        self.inp_shape = inp_shape

        # PSPNet_Model = PSPNet(pretrained=True)

        self.dropout = nn.Dropout2d(0.1)
        self.cls = nn.Conv2d(512, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.cls(x)
        x = F.upsample(x, size=self.inp_shape, mode='bilinear')
        return x



class ImgEncoder(nn.Module):
    def __init__(self, input_dim = 3, dim = 64, n_downsample = 2, n_res = 4, activ='relu', norm = 'in', pad_type='reflect'):
        super(ImgEncoder, self).__init__()

        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim
    def forward(self, x):
        output = self.model(x)
        return output

class ImgEncoder_SPADE(nn.Module):
    def __init__(self, input_dim = 3, dim = 64, n_downsample = 2, n_res = 4, pad_type='reflect',num_classes=19):
        super(ImgEncoder_SPADE, self).__init__()

        self.model = []
        self.model += [SPADEConv2dBlock(input_dim, dim, 3, pad_type=pad_type, num_classes=num_classes)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [SPADEConv2dBlock(dim, 2 * dim, 4, stride=2, pad_type=pad_type, num_classes=num_classes)]
            dim *= 2
        # residual blocks
        self.model += [SPADEResBlocks(n_res, dim, num_classes=num_classes, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim
    def forward(self, x):
        output = self.model(x)
        return output


class ImgDecoder_Naiive(nn.Module):
    def __init__(self, dim = 256, output_dim = 3, n_upsample = 2, n_res = 4, norm='in', activ='relu', pad_type='reflect'):
        super(ImgDecoder_Naiive, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='in', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ImgDecoder_SPADE(nn.Module):
    def __init__(self, dim=256, output_dim=3, n_upsample=2, n_res=4, pad_type='zero', num_classes=19):
        super(ImgDecoder_SPADE, self).__init__()

        self.model = []
        self.model += [SPADEResBlocks(n_res, dim, num_classes=num_classes, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [SPADEConv2dBlock(dim, dim // 2, 5, pw=2, num_classes=num_classes, upsample_in=True)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model = nn.Sequential(*self.model)
        self.conv_rgb = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)

    def forward(self, x):
        #img, segmap = x[0], x[1]
        out_fea = self.model(x)
        return self.conv_rgb(out_fea[0])



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # FCN classification layer
        self.dim = 64
        self.n_layer = 4
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.num_scales = 3
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(3, dim, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        n_classes = pspnet_specs['n_classes']
        # FCN classification layer

        self.feature = nn.Sequential(
            Conv2dBlock(n_classes, 64, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(64, 128, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(128, 256, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(256, 512, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            nn.Conv2d(512, 1, 4, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.feature(x)
        return x



from torch.autograd import Function
class GradientReversalLayer(Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.alpha = lambda_

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        output = grad.neg() * ctx.alpha

        return output, None

class FeatureDomainClassifier(nn.Module):

    def __init__(self, in_channel = 2048, use_dtl = False):
        super(FeatureDomainClassifier, self).__init__()

        self.use_dtl = use_dtl
        self.in_channel = in_channel
        '''
        self.post_process = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=512, stride=2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, stride=2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=1024, stride=2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024), nn.ReLU(True),
            nn.Conv2d(in_channels=1024, out_channels=2048, stride=1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(2048), nn.ReLU(True)
        )
        '''
        self.post_process = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=512, stride=2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=786, stride=2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(786), nn.ReLU(True),
            nn.Conv2d(in_channels=786, out_channels=1024, stride=1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1024), nn.ReLU(True)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, 2)
        self.soft = nn.Softmax(dim=1)

    def forward(self, feature, lambda_=1):

        if not self.use_dtl:
            feature = GradientReversalLayer.apply(feature, lambda_)

        feat = self.post_process(feature)
        feat = self.global_avg_pool(feat)
        feat = feat.view(-1, 1024)
        domain_output = self.classifier(feat)
        domain_output = self.soft(domain_output)

        return domain_output
