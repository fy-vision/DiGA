'''
Misc Utility functions
'''
from collections import OrderedDict
import os
import numpy as np
import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import torch.fft
import kornia

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

def poly_lr_scheduler_warm(base_lr, iter, warmup = 1, max_iter=80000, power=1.0):
    if iter<=warmup:
        return base_lr * (iter / warmup)
    else:
        return base_lr * ((1 - float(iter-warmup) / max_iter) ** (power))


def poly_lr_scheduler(base_lr, iter, max_iter=80000, power=0.9):
	return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(opts, base_lr, i_iter, max_iter, power):
	lr = poly_lr_scheduler(base_lr, i_iter, max_iter, power)
	for opt in opts:
		opt.param_groups[0]['lr'] = lr
		if len(opt.param_groups) > 1:
			opt.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_warm(opts, base_lr, i_iter, max_iter, power):
	lr = poly_lr_scheduler_warm(base_lr, i_iter, max_iter, power)
	for opt in opts:
		opt.param_groups[0]['lr'] = lr
		if len(opt.param_groups) > 1:
			opt.param_groups[1]['lr'] = lr * 10

def generate_mask(imgs, mask_blk_size = 64, mask_ratio = 0.5):
    B, _, H, W = imgs.shape
    mshape = B, 1, round(H / mask_blk_size), round(W / mask_blk_size)
    input_mask = torch.rand(mshape, device=imgs.device)
    input_mask = (input_mask > mask_ratio).float()
    #input_mask = nn.functional.interpolate(input_mask, (H, W), mode='bilinear', align_corners=True)
    input_mask = nn.functional.interpolate(input_mask, (H, W), mode='nearest')
    return input_mask


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images 
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value

    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def save_models(model_dict, prefix='./'):
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    for key, value in model_dict.items():
        torch.save(value.state_dict(), os.path.join(prefix, key+'.pth'))

def load_models(model_dict, prefix='./'):
    for key, value in model_dict.items():
        value.load_state_dict(torch.load(os.path.join(prefix, key+'.pth')))


def create_teacher_params(teacher, student):
    for param in teacher.parameters():
        param.detach_()
    student_param = list(student.parameters())
    teacher_param = list(teacher.parameters())
    n = len(student_param)
    for i in range(0, n):
        teacher_param[i].data[:] = student_param[i].data[:].clone()
    return teacher.cuda()

def update_teacher_params(teacher, student, iteration, stage0=True):
    # Use the "true" average until the exponential average is more correct
    if stage0==True:
        alpha_teacher = min(1 - 1 / (iteration + 1), 0.999)
    else:
        alpha_teacher = 0.999

    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        teacher_param.data[:] = alpha_teacher * teacher_param[:].data[:] + (1 - alpha_teacher) * student_param[:].data[:]
    return teacher.cuda()

def label_one_hot(labels_batch, num_classes=16):
    labels = labels_batch.clone()
    labels[labels == 255] = num_classes
    label_one_hot = torch.nn.functional.one_hot(labels, num_classes + 1).float().cuda()
    label_one_hot = label_one_hot.permute(0, 3, 1, 2)[:, :-1, :, :]
    return label_one_hot


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
        :param tensor: tensor image of size (B,C,H,W) to be un-normalized
        :return: UnNormalized image
        """
        ten = tensor.clone().permute(1, 2, 3, 0)
        for t, m, s in zip(ten, self.mean, self.std):
            t.mul_(s).add_(m)
        return ten.permute(3, 0, 1, 2)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
        :param tensor: tensor image of size (B,C,H,W) to be un-normalized
        :return: UnNormalized image
        """
        ten = tensor.clone().permute(1, 2, 3, 0)
        for t, m, s in zip(ten, self.mean, self.std):
            t.sub_(m).div_(s)
        return ten.permute(3, 0, 1, 2)

def process_label(label, class_numbers=16):
    batch, channel, w, h = label.size()
    pred1 = torch.zeros(batch, class_numbers + 1, w, h).cuda()
    id = torch.where(label < class_numbers, label, torch.Tensor([class_numbers]).cuda())
    pred1 = pred1.scatter_(1, id.long(), 1)
    return pred1

def adaptive_instance_normalization(content_feat, style_feat):
    if content_feat.size()[:2] != style_feat.size()[:2]:
        return content_feat
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def denorm_(img, mean, std):
    return img.mul_(std).add_(mean).div_(255.0)


def renorm_(img, mean, std):
    return img.mul_(255.0).sub_(mean).div_(std)

def strong_transform(param, data=None):
    assert (data is not None)
    data = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data)
    data = gaussian_blur(blur=param['blur'], data=data)
    return data

def color_jitter(color_jitter=None, s=.25, p=.2, mean=None, std=None, data=None):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                data = denorm_(data, mean, std)
                data = seq(data)
                data = renorm_(data, mean, std)
    return data


def gaussian_blur(blur, data=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data



def get_mean_std(batch_size, dev):
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    mean = [
        torch.as_tensor(img_norm_cfg['mean'], device=dev)
        for i in range(batch_size)
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_norm_cfg['std'], device=dev)
        for i in range(batch_size)
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std



def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha

def low_freq_mutate( amp_src, amp_trg, L=0.05 ):
    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src

def fourier_exchange(src_img, trg_img, L=0.01):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    #old version
    #fft_src = torch.rfft( src_img.clone(), signal_ndim=2, onesided=False )
    #fft_trg = torch.rfft( trg_img.clone(), signal_ndim=2, onesided=False )
    #new_version
    fft_src_complex = torch.fft.fftn(src_img, dim=(-2, -1))
    fft_src = torch.stack((fft_src_complex.real, fft_src_complex.imag), -1)
    fft_trg_complex = torch.fft.fftn(trg_img, dim=(-2, -1))
    fft_trg = torch.stack((fft_trg_complex.real, fft_trg_complex.imag), -1)

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )

    # recompose fft of source
    fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    fft_src_[:,:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:,:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    #_, _, imgH, imgW = src_img.size()
    #src_in_trg = torch.irfft( fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )
    src_in_trg = torch.fft.ifftn(torch.complex(fft_src_[..., 0], fft_src_[..., 1]), dim=(-2, -1))

    return src_in_trg.real


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def calc_mean(feat):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    feat_mean = feat_mean.repeat(1, 1, size[2], size[3])
    return feat_mean

def calc_std(feat):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_std = feat.view(N, C, -1).std(dim=2).view(N, C, 1, 1)
    feat_std = feat_std.repeat(1, 1, size[2], size[3])
    return feat_std
