from .model_util import *
from .seg_model_noaux import DeeplabMulti

pspnet_specs = {
    'n_classes': 16,
    'input_size': (713, 713),
    'block_config': [3, 4, 23, 3],
}


class SegModel(nn.Module):
    def __init__(self):
        super(SegModel, self).__init__()
        self.n_classes = pspnet_specs['n_classes']

        model_seg = DeeplabMulti(pretrained=True, num_classes=self.n_classes)

        self.layer0 = nn.Sequential(model_seg.conv1, model_seg.bn1, model_seg.relu, model_seg.maxpool)
        self.layer1 = model_seg.layer1
        self.layer2 = model_seg.layer2
        self.layer3 = model_seg.layer3
        self.layer4 = model_seg.layer4
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
        b.append(self.final.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': 1 * learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]


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
