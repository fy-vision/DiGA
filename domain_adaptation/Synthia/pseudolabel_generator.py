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
from model.model_noaux import SegModel
from util.utils import load_models
from util.loader.CityLoader import CityLoader


CITY_DATA_PATH = './data/Cityscapes'
DATA_LIST_PATH_TRAIN_IMG = './util/loader/cityscapes_list/train.txt'
DATA_LIST_PATH_TRAIN_LBL  = './util/loader/cityscapes_list/train_label.txt'
WEIGHT_DIR = './work_dir/weights'
OUTPUT_DIR = './data/Cityscapes/pseudo_train_warm_up'
CUDA_DIVICE_ID = '0'
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

parser = argparse.ArgumentParser(description='DiGA\
	for domain adaptive semantic segmentation')
parser.add_argument('--weight_dir', type=str, default=WEIGHT_DIR)
parser.add_argument('--city_data_path', type=str, default=CITY_DATA_PATH, help='the path to cityscapes.')
parser.add_argument('--data_list_path_train_img', type=str, default=DATA_LIST_PATH_TRAIN_IMG)
parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
parser.add_argument('--data_list_path_train_lbl', type=str, default=DATA_LIST_PATH_TRAIN_LBL)
parser.add_argument('--cuda_device_id', nargs='+', type=str, default=CUDA_DIVICE_ID)

args = parser.parse_args()

print ('cuda_device_id:', ','.join(args.cuda_device_id))
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.cuda_device_id)


palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 60, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


train_set   = CityLoader(args.city_data_path, args.data_list_path_train_img, args.data_list_path_train_lbl, max_iters=None, crop_size=[1024, 2048], mean=IMG_MEAN, set='train', return_name = True)
train_loader= torch_data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

upsample_1024 = nn.Upsample(size=[1024, 2048], mode='bilinear', align_corners=True)

model_dict = {}

student = SegModel().cuda()
model_dict['student'] = student

load_models(model_dict, args.weight_dir)

student.eval()

predicted_label = np.zeros((len(train_loader), 1024, 2048))
image_name = []

for index, batch in enumerate(train_loader):
    if index % 100 == 0:
        print('%d processd' % index)
    image, _, name = batch
    image_ds = nn.functional.interpolate(image, (512, 1024), mode='bilinear', align_corners=True)
    with torch.no_grad():
        _, _, output_ds, _ = student(Variable(image_ds).cuda())
        _, _, output, _ = student(Variable(image).cuda())
    output = upsample_1024(output)
    output_ds = upsample_1024(output_ds)
    #output = torch.add(output_ds, output) / 2
    output = torch.max(output_ds, output)
    output = nn.functional.softmax(output, dim=1)
    output = output.cpu().data[0].numpy()
    output = output.transpose(1, 2, 0)

    label, _ = np.argmax(output, axis=2), np.max(output, axis=2)
    predicted_label[index] = label.copy()
    image_name.append(name[0])

for index in range(len(train_loader)):
    name = image_name[index]
    label = predicted_label[index]
    output = np.asarray(label, dtype=np.uint8)
    '''
    output = Image.fromarray(output)
    name = name.split('/')[-1]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output.save(os.path.join(args.output_dir, name))
    '''
    output = colorize_mask(output)

    name = name.split('/')[-1]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output.save(os.path.join(args.output_dir, name))
