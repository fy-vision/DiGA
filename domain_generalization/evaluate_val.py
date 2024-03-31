import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as torch_data

import os

# from tensorboardX import SummaryWriter
from PIL import Image
from torch.autograd import Variable

from util.metrics import runningScore
from model.model_noaux import SegModel
from util.utils import load_models
from util.loader.CityLoader import CityLoader
from util.loader.BDDLoader import BDDLoader
from util.loader.MapillaryLoader import MapillaryLoader

num_classes = 19
DATASET_NAME = 'Cityscapes'
DATA_PATH = './data/Cityscapes'
DATA_LIST_PATH_VAL_IMG = './util/loader/cityscapes_list/val.txt'
DATA_LIST_PATH_VAL_LBL  = './util/loader/cityscapes_list/val_label.txt'
WEIGHT_DIR = './work_dir/weights_DiGA_DG'
#DATA_PATH = './data/BDD100k'
#DATA_LIST_PATH_VAL_IMG = './util/loader/bdd100k_list/val.txt'
#DATA_LIST_PATH_VAL_LBL  = './util/loader/bdd100k_list/val_label.txt'
#DATA_PATH = './data/Mapillary'
#DATA_LIST_PATH_VAL_IMG = './util/loader/mapillary_list/val.txt'
#DATA_LIST_PATH_VAL_LBL  = './util/loader/mapillary_list/val_label.txt'
CUDA_DIVICE_ID = '0'

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

parser = argparse.ArgumentParser(description='DiGA for domain generalized semantic segmentation')
parser.add_argument('--weight_dir', type=str, default=WEIGHT_DIR)
parser.add_argument('--data_path', type=str, default=DATA_PATH, help='the path to the data.')
parser.add_argument('--dataset_name', type=str, default=DATASET_NAME, help='the name of the dataset.')
parser.add_argument('--data_list_path_val_img', type=str, default=DATA_LIST_PATH_VAL_IMG)
parser.add_argument('--cuda_device_id', nargs='+', type=str, default=CUDA_DIVICE_ID)
parser.add_argument('--data_list_path_val_lbl', type=str, default=DATA_LIST_PATH_VAL_LBL)

args = parser.parse_args()

print ('cuda_device_id:', ','.join(args.cuda_device_id))
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.cuda_device_id)

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

class InvalidDatasetError(Exception):
    pass


args = parser.parse_args()

if args.dataset_name == 'Cityscapes':
    val_set   = CityLoader(args.data_path, args.data_list_path_val_img, args.data_list_path_val_lbl, max_iters=None, crop_size=[1024, 2048], mean=IMG_MEAN, set='val')
elif args.dataset_name == 'BDD100k':
    val_set   = BDDLoader(args.data_path, args.data_list_path_val_img, args.data_list_path_val_lbl, max_iters=None, crop_size=[720, 1280], mean=IMG_MEAN, set='val')
elif args.dataset_name == 'Mapillary':
    val_set   = MapillaryLoader(args.data_path, args.data_list_path_val_img, args.data_list_path_val_lbl, max_iters=None, crop_size=[1080, 1920], mean=IMG_MEAN, set='validation')
else:
    raise InvalidDatasetError(f'Invalid Dataset Name: {args.dataset_name}')

val_loader= torch_data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

upsample_1024 = nn.Upsample(size=[1024, 2048], mode='bilinear', align_corners=True)
upsample_1080 = nn.Upsample(size=[1080, 1920], mode='bilinear', align_corners=True)
upsample_720 = nn.Upsample(size=[720, 1280], mode='bilinear', align_corners=True)

model_dict = {}

student = SegModel().cuda()
model_dict['student'] = student

load_models(model_dict, args.weight_dir)

student.eval()

cty_running_metrics = runningScore(num_classes)
print('evaluating models ...')
for i_val, (images_val, labels_val) in enumerate(val_loader):
    print(i_val)
    #multi-scale testing
    images_val = Variable(images_val.cuda(), requires_grad=False)
    labels_val = Variable(labels_val, requires_grad=False)

    if args.dataset_name == 'Cityscapes':
        images_ds_val = nn.functional.interpolate(images_val, (512, 1024), mode='bilinear', align_corners=True)
    elif args.dataset_name == 'BDD100k':
        images_ds_val = nn.functional.interpolate(images_val, ([360, 640]), mode='bilinear', align_corners=True)
    elif args.dataset_name == 'Mapillary':
        images_ds_val = nn.functional.interpolate(images_val, (540, 960), mode='bilinear', align_corners=True)
    with torch.no_grad():
        _, _, pred, _ = student(images_val)
        _, _, pred_ds, _= student(images_ds_val)

    if args.dataset_name == 'Cityscapes':
        pred = upsample_1024(pred)
        pred_ds = upsample_1024(pred_ds)
    elif args.dataset_name == 'BDD100k':
        pred = upsample_720(pred)
        pred_ds = upsample_720(pred_ds)
    elif args.dataset_name == 'Mapillary':
        pred = upsample_1080(pred)
        pred_ds = upsample_1080(pred_ds)
    pred = torch.max(pred, pred_ds)
    #pred = torch.add(pred, pred0)/2
    pred = pred.data.max(1)[1].cpu().numpy()
    gt = labels_val.data.cpu().numpy()
    cty_running_metrics.update(gt, pred)
cty_score, cty_class_iou = cty_running_metrics.get_scores()

for k, v in cty_score.items():
    print(k, v)

