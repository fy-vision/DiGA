import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.utils.data as torch_data
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
import os

# from tensorboardX import SummaryWriter
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm
import kornia
import random

from util.loader.CityLoader import CityLoader
from util.loader.SYNTHIALoader import SYNTHIALoader
from util.loader.augmentations import Compose, RandomHorizontallyFlip, RandomSized_and_Crop, Compose_Pseudo, RandomHorizontallyFlip_Pseudo, RandomSized_and_Crop_Pseudo
from util.metrics import runningScore
from util.loss import OhemCrossEntropy, distillation_loss
from calc_centroids import Class_Features
from model.model_noaux import SegModel, ImgEncoder, ImgDecoder
from util.utils import adjust_learning_rate, save_models, load_models, create_teacher_params, update_teacher_params, UnNormalize, Normalize

# Data-related
LOG_DIR = './work_dir/log_DiGA_ST'
CENTROID_DIR = './work_dir/class_centroids/feat_centroids'
GEN_IMG_DIR = './work_dir/generated_imgs_DiGA_ST'
LOAD_WEIGHT_DIR = './work_dir/weights_DiGA_warm_up/'
SAVE_WEIGHT_DIR = './work_dir/weights_DiGA_ST/'
PSEUDO_DIR = 'pseudo_train_warm_up'

SYNTHIA_DATA_PATH = './data/RAND_CITYSCAPES'
CITY_DATA_PATH = './data/Cityscapes'
DATA_LIST_PATH_SYNTHIA = './util/loader/synthia_list/train.txt'
DATA_LIST_PATH_CITY_IMG = './util/loader/cityscapes_list/train.txt'
DATA_LIST_PATH_CITY_LBL = './util/loader/cityscapes_list/train_label.txt'
DATA_LIST_PATH_VAL_IMG = './util/loader/cityscapes_list/val.txt'
DATA_LIST_PATH_VAL_LBL = './util/loader/cityscapes_list/val_label.txt'
CUDA_DIVICE_ID = '0'

parser = argparse.ArgumentParser(description='DiGA for unsupervised domain adaptation for semantic segmentation')
parser.add_argument('--dump_logs', type=bool, default=False)
parser.add_argument('--log_dir', type=str, default=LOG_DIR, help='the path to where you save plots and logs.')
parser.add_argument('--centroid_dir', type=str, default=CENTROID_DIR, help='the path to where you save class centroids.')
parser.add_argument('--load_weight_dir', type=str, default=LOAD_WEIGHT_DIR, help='the path to where you load the model weights.')
parser.add_argument('--save_weight_dir', type=str, default=SAVE_WEIGHT_DIR, help='the path to where you save the model weights.')
parser.add_argument('--pseudo_dir', type=str, default=PSEUDO_DIR, help='the path to where the pseudo labels for the target domain are saved.')
parser.add_argument('--gen_img_dir', type=str, default=GEN_IMG_DIR, help='the path to where you save translated images and segmentation maps.')
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

if not os.path.exists(args.centroid_dir):
    os.makedirs(args.centroid_dir)

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
batch_size_hs = 2
batch_size = batch_size_ls + batch_size_hs

max_epoch = 150
num_steps  = 80000
num_calmIoU = 1000

learning_rate_seg = 2.5e-4
power             = 0.9
weight_decay      = 0.0005
beta = 0.3

lambda_distil = 0.25
lambda_seg = 1.0

# Setup Augmentations
synthia_data_aug = Compose([RandomHorizontallyFlip(),
                         RandomSized_and_Crop([512, 896])
                         ])

city_data_aug = Compose_Pseudo([RandomHorizontallyFlip_Pseudo(),
                         RandomSized_and_Crop_Pseudo([512, 896])
                        ])

extra_aug = nn.Sequential(
                          kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.5, return_transform=False),
                          #kornia.augmentation.RandomSolarize(0.5, 0.5, p=0.2, return_transform=False),
                          kornia.augmentation.RandomGrayscale(p=0.3, return_transform=False),
                          kornia.augmentation.RandomGaussianBlur((3,3),(2,2), p=0.8, return_transform=False),
                          kornia.augmentation.RandomSharpness(0.5, p=0.3, return_transform=False),
                          )
# kornia.augmentation.RandomEqualize(p=0.2, return_transform=False),
# kornia.augmentation.Normalize(mean=0.5*torch.ones(3), std=torch.ones(3), p=1.0, return_transform=False),
# kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.6, return_transform=False)


# ==== DataLoader ====
synthia_set   = SYNTHIALoader(args.synthia_data_path, args.data_list_path_synthia, max_iters=num_steps*batch_size_ls, crop_size=source_input_size, transform=synthia_data_aug, mean=IMG_MEAN)
source_loader = torch_data.DataLoader(synthia_set, batch_size=batch_size_ls, shuffle=True, num_workers=1, pin_memory=True)

synthia_set_full   = SYNTHIALoader(args.synthia_data_path, args.data_list_path_synthia, max_iters=num_steps*batch_size_hs, crop_size=source_input_size_full, transform=synthia_data_aug, mean=IMG_MEAN)
source_loader_full = torch_data.DataLoader(synthia_set_full, batch_size=batch_size_hs, shuffle=True, num_workers=1, pin_memory=True)

city_set   = CityLoader(args.city_data_path, args.data_list_path_city_img, args.data_list_path_city_lbl, max_iters=num_steps* batch_size_ls, crop_size=target_input_size, transform=city_data_aug, mean=IMG_MEAN, set='train', use_pseudo=True, pseudo_dir=args.pseudo_dir)
target_loader= torch_data.DataLoader(city_set, batch_size=batch_size_ls, shuffle=True, num_workers=1, pin_memory=True)

city_set_full   = CityLoader(args.city_data_path, args.data_list_path_city_img, args.data_list_path_city_lbl, max_iters=num_steps* batch_size_hs, crop_size=target_input_size_full, transform=city_data_aug, mean=IMG_MEAN, set='train', use_pseudo=True, pseudo_dir=args.pseudo_dir)
target_loader_full= torch_data.DataLoader(city_set_full, batch_size=batch_size_hs, shuffle=True, num_workers=1, pin_memory=True)

val_set   = CityLoader(args.city_data_path, args.data_list_path_val_img, args.data_list_path_val_lbl, max_iters=None, crop_size=[1024, 2048], mean=IMG_MEAN, set='val')
val_loader= torch_data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


sourceloader_iter_full = enumerate(source_loader_full)
targetloader_iter_full = enumerate(target_loader_full)
sourceloader_iter = enumerate(source_loader)
targetloader_iter = enumerate(target_loader)

# Setup Metrics
cty_running_metrics = runningScore(num_classes)

model_dict = {}

# Setup Model
print ('building models ...')
enc_s      = ImgEncoder().cuda()
dec_s2t      = ImgDecoder().cuda()
model_dict['enc_s'] = enc_s
model_dict['dec_s2t'] = dec_s2t
student = SegModel().cuda()
teacher = SegModel().cuda()


model_dict['student'] = student
model_dict['teacher'] = teacher
load_models(model_dict, args.load_weight_dir)


objective_vectors = torch.load(args.centroid_dir)
class_features = Class_Features(numbers=19)
class_features.objective_vectors = torch.Tensor(objective_vectors).cuda()

student_opt = optim.SGD(student.optim_parameters(learning_rate_seg), lr=learning_rate_seg, momentum=0.9, weight_decay=weight_decay)


seg_opt_list  = []

# Optimizer list for quickly adjusting learning rate
seg_opt_list.append(student_opt)

cudnn.enabled   = True
cudnn.benchmark = False

seg_loss = OhemCrossEntropy().cuda()

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
unnorm = UnNormalize(mean, std)
norm = Normalize(mean, std)

upsample_1024 = nn.Upsample(size=[1024, 2048], mode='bilinear', align_corners=True)
upsample_src = nn.Upsample(size=[512, 896], mode='bilinear', align_corners=True)
upsample_tgt = nn.Upsample(size=[512, 896], mode='bilinear', align_corners=True)


i_iter_tmp  = []
epoch_tmp = []

loss_semseg_tmp = []

City_tmp  = [] 

teacher = create_teacher_params(teacher, student)

student.train()
enc_s.eval()
dec_s2t.eval()
for p in enc_s.parameters():
    p.requires_grad = False
for p in dec_s2t.parameters():
    p.requires_grad = False

best_iou = 0
best_iter= 0
for i_iter in range(num_steps):
    sys.stdout.flush()

    student.train()
    adjust_learning_rate(seg_opt_list , base_lr=learning_rate_seg, i_iter=i_iter, max_iter=num_steps, power=power)
    with torch.no_grad():
        #teacher is the exponentially moving average of the student
        teacher = update_teacher_params(teacher, student, i_iter)

    # ==== sample data ====
    _, source_batch_full = next(sourceloader_iter_full)
    idx_s, source_batch = next(sourceloader_iter)
    idx_t, target_batch = next(targetloader_iter)
    _, target_batch_full = next(targetloader_iter_full)

    source_data_full, source_label_full = source_batch_full
    target_data_full, target_label_full, target_label_pseudo_full = target_batch_full
    source_data, source_label = source_batch
    target_data, target_label, target_label_pseudo = target_batch

    sdatav_full = Variable(source_data_full).cuda()
    slabelv_full = Variable(source_label_full).cuda()
    tdatav_full = Variable(target_data_full).cuda()
    tlabelv_full = Variable(target_label_full).cuda()
    sdatav = Variable(source_data).cuda()
    slabelv = Variable(source_label).cuda()
    tdatav = Variable(target_data).cuda()
    tlabelv = Variable(target_label).cuda()
    tlabelv_pseudo_prob = Variable(target_label_pseudo).cuda()
    tlabelv_pseudo_prob_full = Variable(target_label_pseudo_full).cuda()

    sdatav = torch.cat([sdatav, sdatav_full],dim=0)
    slabelv = torch.cat([slabelv, slabelv_full],dim=0)
    tdatav = torch.cat([tdatav, tdatav_full],dim=0)
    tlabelv_pseudo_prob = torch.cat([tlabelv_pseudo_prob, tlabelv_pseudo_prob_full],dim=0)
    tlabelv = torch.cat([tlabelv, tlabelv_full],dim=0)

    sdatav_aug = beta*norm(extra_aug(sdatav))+(1-beta)*sdatav
    tdatav_aug = beta*norm(extra_aug(tdatav))+(1-beta)*tdatav

    with torch.no_grad():
        feature_s = enc_s(sdatav)
        rec_s2t = dec_s2t(feature_s)

    #Cross-domain Mixture Data Augmentation
    if i_iter >= 0:
        flag = 0
        rec_s2t_clone = rec_s2t.detach().clone()
        sdatav_aug_clone = sdatav_aug.detach().clone()
        mask = torch.zeros(slabelv.size()).cuda()
        for idx in range(slabelv.size()[0]):
            label_list = torch.unique(slabelv[idx]).tolist()
            classes_select = random.sample(label_list, len(label_list) // 2)
            if 255 not in classes_select:
                classes_select.append(255)  # maintaining source don't care region
            for cls_m in classes_select:
                mask[idx][slabelv[idx]==cls_m] = 1
        if not torch.all(torch.eq(slabelv,255)):
            flag = 1
            sdatav_aug_crdomix = torch.zeros(rec_s2t_clone.size()).cuda()
            for idx in range(rec_s2t_clone.size()[0]):
                sdatav_aug_crdomix[idx] = torch.mul(rec_s2t_clone[idx],1-mask[idx]) + torch.mul(sdatav_aug_clone[idx],mask[idx])


    sdatav_cat = torch.cat([sdatav, sdatav_aug_crdomix])

    # forwarding
    _, _, s_pred_cat_stu, s_feat_cat_stu = student(sdatav_cat)

    s_pred_stu = s_pred_cat_stu[:batch_size]
    s_pred_stu_aug = s_pred_cat_stu[batch_size:]

    with torch.no_grad():
        _, _, s_pred_cat_tea, s_feat_cat_tea = teacher(sdatav_cat)
    s_pred_tea_aug_raw = s_pred_cat_tea[batch_size:]
    s_pred_cat_tea = upsample_src(s_pred_cat_tea)

    s_feat_tea_aug = s_feat_cat_tea[batch_size:]
    s_feat_tea = s_feat_cat_tea[:batch_size]

    s_pred_tea = s_pred_cat_tea[:batch_size]
    s_pred_tea_aug = s_pred_cat_tea[batch_size:]

    # bilateral-consensus
    with torch.no_grad():
        tlabelv_pseudo = tlabelv_pseudo_prob.clone()
        _, _, t_pred_tea, t_feat_tea = teacher(tdatav)
        feat_weights = class_features.get_centroid_weight(t_feat_tea.detach())
        feat_weights = upsample_tgt(feat_weights)
        feat_pseudo = feat_weights.max(1, keepdim=True)[1].squeeze(1)
        tlabelv_pseudo[tlabelv_pseudo_prob != feat_pseudo]=255

    if i_iter >= 0:
        #mixing cross domain images for self-training following DACS
        crossmix_label = tlabelv_pseudo.clone()
        tdatav_clone = tdatav_aug.detach().clone()
        sdatav_clone = sdatav.detach().clone()
        mask = torch.zeros(slabelv.size()).cuda()
        for idx in range(slabelv.size()[0]):
            label_list = torch.unique(slabelv[idx]).tolist()
            classes_select = random.sample(label_list, len(label_list) // 2)
            if 255 not in classes_select:
                classes_select.append(255)  # maintaining source don't care region
            #create mask
            for cls_m in classes_select:
                crossmix_label[idx][slabelv[idx]==cls_m] = cls_m
                mask[idx][slabelv[idx]==cls_m] = 1
        if not torch.all(torch.eq(slabelv,255)):
            cross_mix = torch.zeros(tdatav_clone.size()).cuda()
            for idx in range(tdatav_clone.size()[0]):
                cross_mix[idx] = torch.mul(tdatav_clone[idx], 1 - mask[idx]) + torch.mul(sdatav_clone[idx], mask[idx])
            crossmix_label = crossmix_label.long()

    batch, w, h = tlabelv_pseudo.size()
    with torch.no_grad():
        newlabels_t = tlabelv_pseudo.clone().reshape([batch, 1, w, h]).float()
        newlabels_t = F.interpolate(newlabels_t, size=t_feat_tea.size()[2:], mode='nearest')
        vectors_t, ids_t = class_features.calculate_mean_vector(t_feat_tea, t_pred_tea.detach(), newlabels_t)
        # updating class centroids with target images
        for t in range(len(ids_t)):
            class_features.update_objective_SingleVector(ids_t[t], vectors_t[t].detach(), start_mean=False)

        newlabels_s = slabelv.clone().reshape([batch, 1, w, h]).float()
        newlabels_s = F.interpolate(newlabels_s, size=s_feat_tea_aug.size()[2:], mode='nearest')
        vectors_s, ids_s = class_features.calculate_mean_vector(s_feat_tea_aug, s_pred_tea_aug_raw.detach(), newlabels_s)
        #updating class centroids with source images
        for t in range(len(ids_s)):
            class_features.update_objective_SingleVector(ids_s[t], vectors_s[t].detach(), start_mean=False)

    _, _, cross_pred_mix, cross_feat_mix = student(cross_mix)
    cross_pred_mix = upsample_tgt(cross_pred_mix)


    # ==== segmentation loss ====
    s_pred_stu = upsample_src(s_pred_stu)
    loss_semseg = seg_loss(s_pred_stu, slabelv)

    s_pred_cat_stu = upsample_src(s_pred_cat_stu)
    loss_s_distil = distillation_loss(s_pred_cat_tea, s_pred_cat_stu)

    # ==== segmentation loss pseudo target====
    loss_crossmix_semseg = seg_loss(cross_pred_mix, crossmix_label)
    loss_semseg += loss_crossmix_semseg

    # visualize segmentation map
    pred_s = F.softmax(s_pred_stu, dim=1).data.max(1)[1].cpu().numpy()
    pred_t = F.softmax(cross_pred_mix, dim=1).data.max(1)[1].cpu().numpy()

    s_pred_stu_aug = upsample_src(s_pred_stu_aug)
    pred_s_aug = F.softmax(s_pred_stu_aug, dim=1).data.max(1)[1].cpu().numpy()

    map_s  = synthia_set.decode_segmap(pred_s)
    map_t  = city_set.decode_segmap(pred_t)
    map_s_aug  = synthia_set.decode_segmap(pred_s_aug)

    gt_s = slabelv.data.cpu().numpy()
    gt_t = tlabelv.data.cpu().numpy()
    crossmix_pseudo = crossmix_label.data.cpu().numpy()
    pseudo_t_prob = tlabelv_pseudo_prob.data.cpu().numpy()
    pseudo_t = tlabelv_pseudo.data.cpu().numpy()
    pseudo_t_feat = feat_pseudo.data.cpu().numpy()
    gt_s  = synthia_set.decode_segmap(gt_s)
    gt_t  = city_set.decode_segmap(gt_t)
    pseudo_t = city_set.decode_segmap(pseudo_t)
    pseudo_t_prob = city_set.decode_segmap(pseudo_t_prob)
    pseudo_t_feat = city_set.decode_segmap(pseudo_t_feat)
    crossmix_pseudo = city_set.decode_segmap(crossmix_pseudo)

    total_loss = lambda_seg * loss_semseg + lambda_distil * loss_s_distil

    student_opt.zero_grad()
    total_loss.backward()

    student_opt.step()

        
    if i_iter % 50 == 0:
        i_iter_tmp.append(i_iter)
        print('Current Iter : {}/{} , Best Iter : {} , Best mIoU : {}'.format(i_iter, num_steps, best_iter, best_iou))

        plt.title('segmentation_loss')
        loss_semseg_tmp.append(loss_semseg.item())
        plt.plot(i_iter_tmp, loss_semseg_tmp, label='loss_semseg')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'segmentation_loss.png'))
        plt.close()

        plt.title('mIoU')
        plt.plot(epoch_tmp, City_tmp, label='City')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'mIoU.png'))
        plt.close()
        
    if i_iter%100== 0:
        imgs_s = torch.cat(((sdatav[:,[2, 1, 0],:,:].cpu()+1)/2, (sdatav_aug_crdomix[:,[2, 1, 0],:,:].cpu()+1)/2, Variable(torch.Tensor((map_s.transpose((0, 3, 1, 2))))), (sdatav_aug[:,[2, 1, 0],:,:].cpu()+1)/2, (rec_s2t[:,[2, 1, 0],:,:].cpu()+1)/2, Variable(torch.Tensor((map_s_aug.transpose((0, 3, 1, 2))))), Variable(torch.Tensor((gt_s.transpose((0, 3, 1, 2)))))), 0)
        imgs_s = vutils.make_grid(imgs_s.data, nrow=batch_size, normalize=False, scale_each=True).cpu().numpy()
        imgs_s = np.clip(imgs_s*255,0,255).astype(np.uint8)
        imgs_s = imgs_s.transpose(1,2,0)
        imgs_s = Image.fromarray(imgs_s)
        filename = '%05d_source.jpg' % i_iter
        imgs_s.save(os.path.join(args.gen_img_dir, filename))

        imgs_t = torch.cat(((tdatav[:,[2, 1, 0],:,:].cpu()+1)/2, (cross_mix[:,[2, 1, 0],:,:].cpu()+1)/2, Variable(torch.Tensor((crossmix_pseudo.transpose((0, 3, 1, 2))))), Variable(torch.Tensor((map_t.transpose((0, 3, 1, 2))))), (tdatav_aug[:,[2, 1, 0],:,:].cpu()+1)/2, Variable(torch.Tensor((pseudo_t_prob.transpose((0, 3, 1, 2))))), Variable(torch.Tensor((pseudo_t.transpose((0, 3, 1, 2))))), Variable(torch.Tensor((pseudo_t_feat.transpose((0, 3, 1, 2))))), Variable(torch.Tensor((gt_t.transpose((0, 3, 1, 2)))))), 0)
        imgs_t = vutils.make_grid(imgs_t.data, nrow=batch_size, normalize=False, scale_each=True).cpu().numpy()
        imgs_t = np.clip(imgs_t*255,0,255).astype(np.uint8)
        imgs_t = imgs_t.transpose(1,2,0)
        imgs_t = Image.fromarray(imgs_t)
        filename = '%05d_target.jpg' % i_iter
        imgs_t.save(os.path.join(args.gen_img_dir, filename))

    if i_iter % num_calmIoU == 0 and i_iter>0:
        student.eval()
        print ('evaluating models ...')
        for i_val, (images_val, labels_val) in tqdm(enumerate(val_loader)):
            images_val = Variable(images_val.cuda(), requires_grad=False)
            labels_val = Variable(labels_val, requires_grad=False)
            # multi-scale testing including the downscaled image
            image_ds = nn.functional.interpolate(images_val, (512, 1024), mode='bilinear', align_corners=True)
            with torch.no_grad():
                _, _, pred, _ = student(images_val)
                _, _, pred_ds, _ = student(Variable(image_ds).cuda())
            pred = upsample_1024(pred)
            pred_ds = upsample_1024(pred_ds)
            pred = torch.max(pred_ds, pred)
            pred = pred.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()
            cty_running_metrics.update(gt, pred)
            
        cty_score, cty_class_iou = cty_running_metrics.get_scores()
        
        for k, v in cty_score.items():
            print(k, v)
            
        cty_running_metrics.reset()
        City_tmp.append(cty_score['Mean IoU : \t'])
        epoch_tmp.append(i_iter)

        if cty_score['Mean IoU : \t'] > best_iou:
            best_iter = i_iter
            best_iou = cty_score['Mean IoU : \t']
            save_models(model_dict, args.save_weight_dir)

