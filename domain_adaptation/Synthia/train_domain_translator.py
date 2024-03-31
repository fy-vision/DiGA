import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torch.utils.data as torch_data
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
import os

# from tensorboardX import SummaryWriter
from PIL import Image
from torch.autograd import Variable

from util.loader.CityLoader import CityLoader
from util.loader.SYNTHIALoader import SYNTHIALoader
from util.loader.augmentations import Compose, RandomHorizontallyFlip, RandomCrop
from util.metrics import runningScore
from util.loss import cross_entropy2d, seg_edge_loss, VGGLoss
from model.model_noaux import ImgEncoder, ImgDecoder, Discriminator
from util.utils import adjust_learning_rate, save_models

# Data-related
LOG_DIR = './work_dir/log_domain_translator'
GEN_IMG_DIR = './work_dir/generated_imgs_domain_translator'
WEIGHT_DIR = './work_dir/weights_DiGA_domain_translator/'

SYNTHIA_DATA_PATH = './data/RAND_CITYSCAPES'
CITY_DATA_PATH = './data/Cityscapes'
DATA_LIST_PATH_SYNTHIA = './util/loader/synthia_list/train.txt'
DATA_LIST_PATH_CITY_IMG = './util/loader/cityscapes_list/train.txt'
DATA_LIST_PATH_CITY_LBL = './util/loader/cityscapes_list/train_label.txt'
DATA_LIST_PATH_VAL_IMG = './util/loader/cityscapes_list/val.txt'
DATA_LIST_PATH_VAL_LBL = './util/loader/cityscapes_list/val_label.txt'
# Hyper-parameters
CUDA_DIVICE_ID = '0'

parser = argparse.ArgumentParser(description='Domain translator for DiGA')
parser.add_argument('--dump_logs', type=bool, default=False)
parser.add_argument('--log_dir', type=str, default=LOG_DIR, help='the path to where you save plots and logs.')
parser.add_argument('--gen_img_dir', type=str, default=GEN_IMG_DIR, help='the path to where you save translated images.')
parser.add_argument('--weight_dir', type=str, default=WEIGHT_DIR, help='the path to where you save the model weights.')
parser.add_argument('--synthia_data_path', type=str, default=SYNTHIA_DATA_PATH, help='the path to SYNTHIA dataset.')
parser.add_argument('--city_data_path', type=str, default=CITY_DATA_PATH, help='the path to Cityscapes dataset.')
parser.add_argument('--data_list_path_synthia', type=str, default=DATA_LIST_PATH_SYNTHIA)
parser.add_argument('--data_list_path_city_img', type=str, default=DATA_LIST_PATH_CITY_IMG)
parser.add_argument('--data_list_path_city_lbl', type=str, default=DATA_LIST_PATH_CITY_LBL)
parser.add_argument('--data_list_path_val_img', type=str, default=DATA_LIST_PATH_VAL_IMG)
parser.add_argument('--data_list_path_val_lbl', type=str, default=DATA_LIST_PATH_VAL_LBL)
parser.add_argument('--cuda_device_id', nargs='+', type=str, default=CUDA_DIVICE_ID)

args = parser.parse_args()

print ('cuda_device_id:', ','.join(args.cuda_device_id))
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.cuda_device_id)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
    
if not os.path.exists(args.gen_img_dir):
    os.makedirs(args.gen_img_dir)

if args.dump_logs == True:
	old_output = sys.stdout
	sys.stdout = open(os.path.join(args.log_dir, 'output.txt'), 'w')

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

num_classes = 16
source_input_size_full = [1140, 1920]
target_input_size_full = [1024, 2048]
source_input_size = [760, 1280]
target_input_size = [512, 1024]
batch_size_ls = 1
batch_size_hs = 1
batch_size = batch_size_ls + batch_size_hs

max_epoch = 150
num_steps  = 50000

learning_rate_enc   = 1e-4
learning_rate_dec = 1e-4
learning_rate_dis = 1e-4
power             = 0.9
weight_decay      = 0.0005

lambda_cyc = 10.0
lambda_adv = 0.5
lambda_percep = 0.1
lambda_self = 0.025
lambda_seg_edge = 20.0

# Setup Augmentations
synthia_data_aug = Compose([RandomHorizontallyFlip(),
                            RandomCrop([512, 896])
                            ])

city_data_aug = Compose([RandomHorizontallyFlip(),
                         RandomCrop([512, 896])
                        ])
# ==== DataLoader ====
synthia_set   = SYNTHIALoader(args.synthia_data_path, args.data_list_path_synthia, max_iters=num_steps*batch_size_ls, crop_size=source_input_size, transform=synthia_data_aug, mean=IMG_MEAN)
source_loader = torch_data.DataLoader(synthia_set, batch_size=batch_size_ls, shuffle=True, num_workers=1, pin_memory=True)

synthia_set_full   = SYNTHIALoader(args.synthia_data_path, args.data_list_path_synthia, max_iters=num_steps*batch_size_hs, crop_size=source_input_size_full, transform=synthia_data_aug, mean=IMG_MEAN)
source_loader_full = torch_data.DataLoader(synthia_set_full, batch_size=batch_size_hs, shuffle=True, num_workers=1, pin_memory=True)

city_set   = CityLoader(args.city_data_path, args.data_list_path_city_img, args.data_list_path_city_lbl, max_iters=num_steps*batch_size_ls, crop_size=target_input_size, transform=city_data_aug, mean=IMG_MEAN, set='train')
target_loader= torch_data.DataLoader(city_set, batch_size=batch_size_ls, shuffle=True, num_workers=1, pin_memory=True)

city_set_full   = CityLoader(args.city_data_path, args.data_list_path_city_img, args.data_list_path_city_lbl, max_iters=num_steps*batch_size_hs, crop_size=target_input_size_full, transform=city_data_aug, mean=IMG_MEAN, set='train')
target_loader_full= torch_data.DataLoader(city_set_full, batch_size=batch_size_hs, shuffle=True, num_workers=1, pin_memory=True)

val_set   = CityLoader(args.city_data_path, args.data_list_path_val_img, args.data_list_path_val_lbl, max_iters=None, crop_size=[1024, 2048], mean=IMG_MEAN, set='val')
val_loader= torch_data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

sourceloader_iter = enumerate(source_loader)
targetloader_iter = enumerate(target_loader)
sourceloader_iter_full = enumerate(source_loader_full)
targetloader_iter_full = enumerate(target_loader_full)

# Setup Metrics
cty_running_metrics = runningScore(num_classes)

model_dict = {}

# Setup Model
print ('building models ...')
enc_s      = ImgEncoder().cuda()
enc_t      = ImgEncoder().cuda()

dec_s2t      = ImgDecoder().cuda()
dec_t2s      = ImgDecoder().cuda()
dis_s2t    = Discriminator().cuda()
dis_t2s    = Discriminator().cuda()

model_dict['enc_s'] = enc_s
#model_dict['enc_t'] = enc_t

model_dict['dec_s2t'] = dec_s2t
#model_dict['dec_t2s'] = dec_t2s
#model_dict['dis_s2t'] = dis_s2t
#model_dict['dis_t2s'] = dis_t2s

enc_s_opt = optim.Adam(enc_s.parameters(), lr=learning_rate_enc, betas=(0.5, 0.999))
enc_t_opt = optim.Adam(enc_t.parameters(), lr=learning_rate_enc, betas=(0.5, 0.999))

dec_s2t_opt = optim.Adam(dec_s2t.parameters(), lr=learning_rate_dec, betas=(0.5, 0.999))
dec_t2s_opt = optim.Adam(dec_t2s.parameters(), lr=learning_rate_dec, betas=(0.5, 0.999))
dis_s2t_opt = optim.Adam(dis_s2t.parameters(), lr=learning_rate_dis, betas=(0.5, 0.999))
dis_t2s_opt = optim.Adam(dis_t2s.parameters(), lr=learning_rate_dis, betas=(0.5, 0.999))

seg_opt_list  = []
enc_opt_list = []
dec_opt_list  = []
dis_opt_list  = []

# Optimizer list for quickly adjusting learning rate
enc_opt_list.append(enc_s_opt)
enc_opt_list.append(enc_t_opt)

dec_opt_list.append(dec_s2t_opt)
dec_opt_list.append(dec_t2s_opt)
dis_opt_list.append(dis_s2t_opt)
dis_opt_list.append(dis_t2s_opt)


cudnn.enabled   = True
cudnn.benchmark = True

mse_loss = nn.MSELoss(size_average=True).cuda()
sg_loss  = cross_entropy2d
L1Loss = nn.L1Loss().cuda()
VGG_loss = VGGLoss()

true_label = 1
fake_label = 0

i_iter_tmp  = []
epoch_tmp = []

loss_recon_s_tmp  = []
loss_recon_t_tmp  = []
loss_cyc_s_tmp = []
loss_cyc_t_tmp = []


prob_dis_s2t_real1_tmp = []
prob_dis_s2t_fake1_tmp = []
prob_dis_s2t_fake2_tmp = []
prob_dis_t2s_real1_tmp = []
prob_dis_t2s_fake1_tmp = []
prob_dis_t2s_fake2_tmp = []

City_tmp  = [] 

enc_s.train()
enc_t.train()

dec_s2t.train()
dec_t2s.train()
dis_s2t.train()
dis_t2s.train()

for i_iter in range(num_steps):
    sys.stdout.flush()

    adjust_learning_rate(enc_opt_list, base_lr=learning_rate_enc, i_iter=i_iter, max_iter=num_steps, power=power)
    adjust_learning_rate(dec_opt_list , base_lr=learning_rate_dec, i_iter=i_iter, max_iter=num_steps, power=power)
    adjust_learning_rate(dis_opt_list , base_lr=learning_rate_dis, i_iter=i_iter, max_iter=num_steps, power=power)

    # ==== sample data ====
    _, source_batch_full = next(sourceloader_iter_full)
    idx_s, source_batch = next(sourceloader_iter)
    idx_t, target_batch = next(targetloader_iter)
    _, target_batch_full = next(targetloader_iter_full)

    source_data_full, source_label_full = source_batch_full
    target_data_full, _ = target_batch_full
    source_data, source_label = source_batch
    target_data, _ = target_batch

    sdatav_full = Variable(source_data_full).cuda()
    slabelv_full = Variable(source_label_full).cuda()
    tdatav_full = Variable(target_data_full).cuda()
    sdatav = Variable(source_data).cuda()
    slabelv = Variable(source_label).cuda()
    tdatav = Variable(target_data).cuda()


    sdatav = torch.cat([sdatav, sdatav_full],dim=0)
    slabelv = torch.cat([slabelv, slabelv_full],dim=0)
    tdatav = torch.cat([tdatav, tdatav_full],dim=0)

    feature_s = enc_s(sdatav)
    feature_t = enc_t(tdatav)

    rec_s2t = dec_s2t(feature_s)
    rec_t2s = dec_t2s(feature_t)

    rec_s = dec_t2s(feature_s)
    rec_t = dec_s2t(feature_t)

    feature_s2t = enc_t(rec_s2t)
    feature_t2s = enc_s(rec_t2s)
    rec_s_cyc = dec_t2s(feature_s2t)
    rec_t_cyc = dec_s2t(feature_t2s)

    for p in dis_s2t.parameters():
        p.requires_grad = True
    for p in dis_t2s.parameters():
        p.requires_grad = True

    # train image discriminator -> LSGAN
    # ===== dis_s2t =====
    #if i_iter%1 == 0:
    prob_dis_s2t_real1_list = dis_s2t(tdatav)
    prob_dis_s2t_fake1_list = dis_s2t(rec_s2t.detach())
    loss_d_s2t = 0
    for it, (prob_dis_s2t_real1, prob_dis_s2t_fake1) in enumerate(zip(prob_dis_s2t_real1_list, prob_dis_s2t_fake1_list)):
        #loss_d_s2t += (mse_loss(prob_dis_s2t_real1, Variable(torch.FloatTensor(prob_dis_s2t_real1.data.size()).fill_(true_label).cuda())).cuda() + mse_loss(prob_dis_s2t_fake1, Variable(torch.FloatTensor(prob_dis_s2t_fake1.data.size()).fill_(fake_label).cuda())).cuda())
        loss_d_s2t += torch.mean((prob_dis_s2t_real1 - true_label) ** 2) + torch.mean((prob_dis_s2t_fake1 - fake_label) ** 2)
    dis_s2t_opt.zero_grad()
    loss_d_s2t.backward()
    dis_s2t_opt.step()

    # ===== dis_t2s =====
    #if i_iter%1 == 0:
    prob_dis_t2s_real1_list = dis_t2s(sdatav)
    prob_dis_t2s_fake1_list = dis_t2s(rec_t2s.detach())
    loss_d_t2s = 0
    for it, (prob_dis_t2s_real1, prob_dis_t2s_fake1) in enumerate(zip(prob_dis_t2s_real1_list, prob_dis_t2s_fake1_list)):
        #loss_d_t2s += (mse_loss(prob_dis_t2s_real1, Variable(torch.FloatTensor(prob_dis_t2s_real1.data.size()).fill_(true_label).cuda())).cuda() + mse_loss(prob_dis_t2s_fake1, Variable(torch.FloatTensor(prob_dis_t2s_fake1.data.size()).fill_(fake_label).cuda())).cuda())
        loss_d_t2s += torch.mean((prob_dis_t2s_real1 - true_label) ** 2) + torch.mean((prob_dis_t2s_fake1 - fake_label) ** 2)
    dis_t2s_opt.zero_grad()
    loss_d_t2s.backward()
    dis_t2s_opt.step()

    for p in dis_s2t.parameters():
        p.requires_grad = False
    for p in dis_t2s.parameters():
        p.requires_grad = False
        
    # ==== Image self-reconstruction loss & cycle-reconstruction loss & feature reconstruction loss ====
    loss_recon_s = L1Loss(rec_s, sdatav)
    loss_recon_t = L1Loss(rec_t, tdatav)
    loss_recon_self = loss_recon_s + loss_recon_t

    loss_perceptual  = VGG_loss(rec_s2t, sdatav) + VGG_loss(rec_t2s, tdatav)

    loss_seg_edge_s2t, _, _ = seg_edge_loss(rec_s2t, sdatav, slabelv)

    loss_cyc_s = L1Loss(rec_s_cyc, sdatav)
    loss_cyc_t = L1Loss(rec_t_cyc, tdatav)
    loss_recon_cyc = loss_cyc_s + loss_cyc_t

    # ==== image translation loss ====
    # prob_dis_s2t_real2_list = dis_s2t(tdatav)
    prob_dis_s2t_fake2_list = dis_s2t(rec_s2t)
    loss_gen_s2t = 0
    for it, (prob_dis_s2t_fake2) in enumerate(prob_dis_s2t_fake2_list):
        #loss_gen_s2t += mse_loss(prob_dis_s2t_fake2, Variable(torch.FloatTensor(prob_dis_s2t_fake2.data.size()).fill_(true_label)).cuda()) \
        loss_gen_s2t += torch.mean((prob_dis_s2t_fake2 - true_label) ** 2)

    # prob_dis_t2s_real2_list = dis_t2s(sdatav)
    prob_dis_t2s_fake2_list = dis_t2s(rec_t2s)
    loss_gen_t2s = 0
    for it, (prob_dis_t2s_fake2) in enumerate(prob_dis_t2s_fake2_list):
        #loss_gen_t2s += mse_loss(prob_dis_t2s_fake2, Variable(torch.FloatTensor(prob_dis_t2s_fake2.data.size()).fill_(true_label)).cuda()) \
        loss_gen_t2s += torch.mean((prob_dis_t2s_fake2 - true_label) ** 2)
    loss_image_translation = loss_gen_s2t + loss_gen_t2s

    
    # visualize segmentation map
    total_loss = \
            + lambda_adv * loss_image_translation \
            + lambda_cyc * loss_recon_cyc \
            + lambda_seg_edge * loss_seg_edge_s2t \
            + lambda_percep * loss_perceptual \
            + lambda_self* loss_recon_self

    enc_s_opt.zero_grad()
    enc_t_opt.zero_grad()
    dec_s2t_opt.zero_grad()
    dec_t2s_opt.zero_grad()

    total_loss.backward()

    enc_s_opt.step()
    enc_t_opt.step()
    dec_s2t_opt.step()
    dec_t2s_opt.step()
        
    if i_iter % 50 == 0:
        i_iter_tmp.append(i_iter)
        print('Current Iter : {}/{}'.format(i_iter, num_steps))
        plt.title('prob_s2t')
        prob_dis_s2t_real1_tmp.append(prob_dis_s2t_real1.data[0].mean().cpu())
        prob_dis_s2t_fake1_tmp.append(prob_dis_s2t_fake1.data[0].mean().cpu())
        prob_dis_s2t_fake2_tmp.append(prob_dis_s2t_fake2.data[0].mean().cpu())
        #print('prob_dis_s2t_real1_tmp_length:', len(prob_dis_s2t_real1_tmp))
        #print('i_iter_tmp_length:', len(i_iter_tmp))
        plt.plot(i_iter_tmp, prob_dis_s2t_real1_tmp, label='prob_dis_s2t_real1')
        plt.plot(i_iter_tmp, prob_dis_s2t_fake1_tmp, label='prob_dis_s2t_fake1')
        plt.plot(i_iter_tmp, prob_dis_s2t_fake2_tmp, label='prob_dis_s2t_fake2')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'prob_s2t.png'))
        plt.close()

        plt.title('prob_t2s')
        prob_dis_t2s_real1_tmp.append(prob_dis_t2s_real1.data[0].mean().cpu())
        prob_dis_t2s_fake1_tmp.append(prob_dis_t2s_fake1.data[0].mean().cpu())
        prob_dis_t2s_fake2_tmp.append(prob_dis_t2s_fake2.data[0].mean().cpu())
        plt.plot(i_iter_tmp, prob_dis_t2s_real1_tmp, label='prob_dis_t2s_real1')
        plt.plot(i_iter_tmp, prob_dis_t2s_fake1_tmp, label='prob_dis_t2s_fake1')
        plt.plot(i_iter_tmp, prob_dis_t2s_fake2_tmp, label='prob_dis_t2s_fake2')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'prob_t2s.png'))
        plt.close()

        plt.title('recon cyc loss')
        loss_cyc_s_tmp.append(loss_cyc_s.item())
        loss_cyc_t_tmp.append(loss_cyc_t.item())
        plt.plot(i_iter_tmp, loss_cyc_s_tmp, label='loss_cyc_s')
        plt.plot(i_iter_tmp, loss_cyc_t_tmp, label='loss_cyc_t')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'cyc_recon_loss.png'))
        plt.close()
        
    if i_iter%1000 == 0 :
        imgs_s = torch.cat(((sdatav[:,[2, 1, 0],:,:].cpu()+1)/2, (rec_s_cyc[:,[2, 1, 0],:,:].cpu()+1)/2, (rec_s2t[:,[2, 1, 0],:,:].cpu()+1)/2), 0)
        imgs_s = vutils.make_grid(imgs_s.data, nrow=batch_size, normalize=False, scale_each=True).cpu().numpy()
        imgs_s = np.clip(imgs_s*255,0,255).astype(np.uint8)
        imgs_s = imgs_s.transpose(1,2,0)
        imgs_s = Image.fromarray(imgs_s)
        filename = '%05d_source.jpg' % i_iter
        imgs_s.save(os.path.join(args.gen_img_dir, filename))


        imgs_t = torch.cat(((tdatav[:,[2, 1, 0],:,:].cpu()+1)/2, (rec_t_cyc[:,[2, 1, 0],:,:].cpu()+1)/2, (rec_t2s[:,[2, 1, 0],:,:].cpu()+1)/2), 0)
        imgs_t = vutils.make_grid(imgs_t.data, nrow=batch_size, normalize=False, scale_each=True).cpu().numpy()
        imgs_t = np.clip(imgs_t*255,0,255).astype(np.uint8)
        imgs_t = imgs_t.transpose(1,2,0)
        imgs_t = Image.fromarray(imgs_t)
        filename = '%05d_target.jpg' % i_iter
        imgs_t.save(os.path.join(args.gen_img_dir, filename))

        epoch_tmp.append(i_iter)
        if i_iter % 2000 == 0 and i_iter != 0:
        	save_models(model_dict, args.weight_dir)
