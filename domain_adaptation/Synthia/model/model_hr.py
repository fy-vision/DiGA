from .model_util import *
from .networks import HRNet, OCRNet
import yaml
import os

pspnet_specs = {
    'n_classes': 16,
    'input_size': (713, 713),
    'block_config': [3, 4, 23, 3],
}

# define path to hrnet ocrnet configuration
with open(os.path.join('./model/config_files/config_default.yaml'), 'r') as cfg:
    config = yaml.load(cfg)

class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()
        self.n_classes = pspnet_specs['n_classes']

        config['OCRNET_MODEL']['NUM_CLASSES'] = self.n_classes

        self.hrnet = HRNet(config)
        self.ocrnet = OCRNet(config)

    def forward(self, x):

        shared_shallow = self.hrnet(x)
        pred1, pred2, shared_seg = self.ocrnet(shared_shallow)
        #print('shared_shallow shape',shared_shallow.size())
        #print('shared_seg shape', shared_seg.size()) 512

        return shared_shallow, pred1, pred2, shared_seg
    '''
    def get_1x_lr_params_NOscale(self):
        b = []
        b.append(self.hrnet)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        b = []
        b.append(self.ocrnet.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': 1 * learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]
    '''


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


class ImgDecoder(nn.Module):
    def __init__(self, dim = 256, output_dim = 3, n_upsample = 2, n_res = 4, norm='in', activ='relu', pad_type='reflect'):
        super(ImgDecoder, self).__init__()

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
