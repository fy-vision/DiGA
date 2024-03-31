import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
import os
from torch.autograd import Variable
from util.loader.CityLoader import CityLoader
from util.loader.augmentations import Compose, RandomHorizontallyFlip, RandomCrop
from model.model_hr import SegModel
from util.utils import load_models, process_label


def calc_centroids(opt, model, source_loader, source_loader_full):

    class_features = Class_Features(numbers=19)
    epoch = 5
    opt.source = True

    # begin training
    for epoch in range(epoch):
        model.eval()
        #print('source:',opt.source)
        if opt.source: #source
            for index, (batch, batch_full) in enumerate(zip(source_loader, source_loader_full)):
                if index % 100 == 0:
                    print('epoch', epoch)
                    print('%d processd' % index)
                source_data, source_label = batch
                sdatav = Variable(source_data).cuda()
                #slabelv = Variable(source_label).cuda()
                source_data_full, source_label_full = batch_full
                sdatav_full = Variable(source_data_full).cuda()
                #slabelv_full = Variable(source_label_full).cuda()
                sdatav = torch.cat([sdatav, sdatav_full], dim=0)
                #slabelv = torch.cat([slabelv, slabelv_full], dim=0)

                _, _, out, feature_s = model(sdatav[:,[2, 1, 0],:,:])
                #batch, w, h = slabelv.size()
                #newlabels = slabelv.reshape([batch, 1, w, h]).float()
                #newlabels = F.interpolate(newlabels, size=out.size()[2:], mode='nearest')
                #vectors, ids = class_features.calculate_mean_vector(feature_s, out, newlabels, model)
                vectors, ids = class_features.calculate_mean_vector(feature_s, out, model=model)
                for t in range(len(ids)):
                    class_features.update_objective_SingleVector(ids[t], vectors[t].detach().cpu().numpy(), 'mean')

        save_path = os.path.join(os.path.dirname(opt.centroid_dir), "feat_centroids")
        torch.save(class_features.objective_vectors, save_path)


class Class_Features:
    def __init__(self, numbers = 19):
        self.class_numbers = numbers
        self.class_features = [[] for i in range(self.class_numbers)]
        self.num = np.zeros(numbers)

        self.objective_vectors = torch.zeros([self.class_numbers, 512])
        self.objective_vectors_num = torch.zeros([self.class_numbers])
        self.centroid_momentum = 0.0001
        self.valid_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        self.smooth_loss = nn.SmoothL1Loss()
        self.mse_loss = nn.MSELoss(size_average=True).cuda()

    def calculate_mean_vector_by_output(self, feat_cls, outputs):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = process_label(outputs_argmax.float())
        outputs_pred = outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 5:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                # if (torch.sum(outputs_pred[n][t] * labels_expanded[n][t]).item() < 30):
                #     continue
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                # self.update_cls_feature(vector=s, id=t)
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def calculate_mean_vector(self, feat_cls, outputs, labels_val=None, model=None):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = process_label(outputs_argmax.float())
        if labels_val is None:
            outputs_pred = outputs_argmax
        else:
            labels_expanded = process_label(labels_val)
            outputs_pred = labels_expanded * outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 5:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                # if (torch.sum(outputs_pred[n][t] * labels_expanded[n][t]).item() < 30):
                #     continue
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                # self.update_cls_feature(vector=s, id=t)
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def update_objective_SingleVector(self, id, vector, name='moving_average', start_mean=True):
        if vector.sum().item() == 0:
            return
        if start_mean and self.objective_vectors_num[id].item() < 100:
            name = 'mean'
        if name == 'moving_average':
            self.objective_vectors[id] = self.objective_vectors[id] * (
                    1 - self.centroid_momentum) + self.centroid_momentum * vector.squeeze()
            self.objective_vectors_num[id] += 1
            self.objective_vectors_num[id] = min(self.objective_vectors_num[id], 3000)
        elif name == 'mean':
            self.objective_vectors[id] = self.objective_vectors[id] * self.objective_vectors_num[id] + vector.squeeze()
            self.objective_vectors_num[id] += 1
            self.objective_vectors[id] = self.objective_vectors[id] / self.objective_vectors_num[id]
            self.objective_vectors_num[id] = min(self.objective_vectors_num[id], 3000)
            pass
        else:
            raise NotImplementedError('no such updating way of objective vectors {}'.format(name))

    def feat_centroid_distance(self, feat):
        N, C, H, W = feat.shape
        feat_centroid_distance = -torch.ones((N, self.class_numbers, H, W)).cuda()
        for i in range(self.class_numbers):
            feat_centroid_distance[:, i, :, :] = torch.norm(self.objective_vectors[i].detach().reshape(-1,1,1).expand(-1, H, W) - feat, 2, dim=1)
        return feat_centroid_distance

    def get_centroid_weight(self, feat):
        feat_centroid_distance = self.feat_centroid_distance(feat)
        weight = F.softmax(-feat_centroid_distance, dim=1)
        return weight

    def get_centroid_distance(self, feat):
        feat_centroid_distance = self.feat_centroid_distance(feat)
        return -feat_centroid_distance



if __name__ == "__main__":
# Data-related
    WEIGHT_DIR = './work_dir/weights_DiGA_semiseg_1_16_warm_up/'
    CENTROID_DIR = './work_dir/class_centroids_1_16/'
    CITY_DATA_PATH = './data/Cityscapes'
    # we load all city images, but only use the model prediction to calculate class centroids. GT will never be touched.
    DATA_LIST_PATH_CITY_IMG = './util/loader/cityscapes_list/train.txt'
    DATA_LIST_PATH_CITY_LBL = './util/loader/cityscapes_list/train_label.txt'
    CUDA_DIVICE_ID = '0'

    parser = argparse.ArgumentParser(description='DiGA for semi-supervised semantic segmentation')
    parser.add_argument('--source', type=bool, default=False, help='calculate centroids on the source or target domain')
    parser.add_argument('--weight_dir', type=str, default=WEIGHT_DIR)
    parser.add_argument('--centroid_dir', type=str, default=CENTROID_DIR, help='the path to where you save class centroids.')
    parser.add_argument('--city_data_path', type=str, default=CITY_DATA_PATH, help='the path to Cityscapes dataset.')
    parser.add_argument('--data_list_path_city_img', type=str, default=DATA_LIST_PATH_CITY_IMG)
    parser.add_argument('--data_list_path_city_lbl', type=str, default=DATA_LIST_PATH_CITY_LBL)
    parser.add_argument('--cuda_device_id', nargs='+', type=str, default=CUDA_DIVICE_ID)

    args = parser.parse_args()

    if not os.path.exists(args.centroid_dir):
        os.makedirs(args.centroid_dir)

    print('cuda_device_id:', ','.join(args.cuda_device_id))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.cuda_device_id)

    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    num_classes = 19
    target_input_size = [512, 1024]
    target_input_size_full = [1024, 2048]
    batch_size = 1
    max_epoch = 150
    num_steps = 100000

    # ==== DataLoader ====
    city_data_aug = Compose([RandomHorizontallyFlip(),
                             RandomCrop([512, 1024])
                            ])
    city_set = CityLoader(args.city_data_path, args.data_list_path_city_img, args.data_list_path_city_lbl,max_iters=None, crop_size=target_input_size, transform=city_data_aug, mean=IMG_MEAN, set='train')
    source_loader = torch_data.DataLoader(city_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    city_set_full   = CityLoader(args.city_data_path, args.data_list_path_city_img, args.data_list_path_city_lbl, max_iters=num_steps* batch_size, crop_size=target_input_size_full, transform=city_data_aug, mean=IMG_MEAN, set='train')
    source_loader_full= torch_data.DataLoader(city_set_full, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    model_dict = {}

    # Setup Model
    print('building models ...')
    student = SegModel().cuda()
    model_dict['student'] = student
    load_models(model_dict, args.weight_dir)
    calc_centroids(args, student, source_loader, source_loader_full)

#python calc_centroids.py
