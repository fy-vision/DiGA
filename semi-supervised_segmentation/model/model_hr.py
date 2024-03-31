from .model_util import *
#from .seg_model import DeeplabMulti

from .networks import HRNet, OCRNet
import yaml
import os

pspnet_specs = {
    'n_classes': 19,
    'input_size': (713, 713),
    'block_config': [3, 4, 23, 3],
}

# define path to hrnet ocrnet configuration
with open(os.path.join('./model/config_files/config_default.yaml'), 'r') as cfg:
    config = yaml.load(cfg)

class SegModel(nn.Module):
    def __init__(self):
        super(SegModel, self).__init__()
        self.n_classes = pspnet_specs['n_classes']

        config['OCRNET_MODEL']['NUM_CLASSES'] = self.n_classes

        self.hrnet = HRNet(config)
        self.ocrnet = OCRNet(config)

    def forward(self, x):

        shared_shallow = self.hrnet(x)
        pred1, pred2, shared_seg = self.ocrnet(shared_shallow)

        return shared_shallow, pred1, pred2, shared_seg



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


