import cv2
import kornia
import matplotlib

matplotlib.use('TkAgg')

import os
import numpy as np
import torch

from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import tqdm

# from model.model import SharedEncoder
from util.metrics import runningScore
from util.utils import load_models, colorize_mask

import matplotlib.pyplot as plt

methods = ['Source_Only', 'BDL', 'ProDA', 'CDBP_576']
shortlisted_im_names = ['frankfurt_000000_001016_leftImg8bit.png', 'frankfurt_000000_008451_leftImg8bit.png',
                        'frankfurt_000000_019607_leftImg8bit.png', 'frankfurt_000001_007285_leftImg8bit.png',
                        'frankfurt_000001_062793_leftImg8bit.png', 'frankfurt_000001_071781_leftImg8bit.png',
                        'frankfurt_000001_073464_leftImg8bit.png', 'lindau_000007_000019_leftImg8bit.png',
                        'munster_000000_000019_leftImg8bit.png', 'munster_000001_000019_leftImg8bit.png',
                        'munster_000003_000019_leftImg8bit.png', 'munster_000012_000019_leftImg8bit.png',
                        'munster_000025_000019_leftImg8bit.png', 'munster_000048_000019_leftImg8bit.png',
                        'munster_000055_000019_leftImg8bit.png', 'munster_000103_000019_leftImg8bit.png',
                        'munster_000129_000019_leftImg8bit.png', 'munster_000166_000019_leftImg8bit.png',
                        'munster_000168_000019_leftImg8bit.png']
shortlisted_im_names = ['frankfurt_000000_008451_leftImg8bit.png',
                        'frankfurt_000001_073464_leftImg8bit.png',
                        'munster_000003_000019_leftImg8bit.png',
                        'munster_000048_000019_leftImg8bit.png',
                        'munster_000103_000019_leftImg8bit.png',
                        'munster_000129_000019_leftImg8bit.png',
                        'munster_000166_000019_leftImg8bit.png']


def resize_anyto1024x512(im_path):
    im_pil = Image.open(im_path).convert('RGB')
    im_t = torch.tensor(np.array(im_pil)).permute(2, 0, 1).unsqueeze(0)
    return F.interpolate(im_t, (512, 1024), mode='nearest').squeeze(0).permute(1, 2, 0)


def load_images_folder(base_path, shortlist_im_num):
    data, list_of_images = {}, []
    num_classes = 19
    cty_running_metrics = {}
    for method in methods:
        cty_running_metrics[method] = {}
        cty_running_metrics[method]['whole'] = runningScore(num_classes)
        cty_running_metrics[method]['edges'] = runningScore(num_classes)
        cty_running_metrics[method]['non_edges'] = runningScore(num_classes)

    for iter_l, im_name in tqdm(enumerate(sorted(os.listdir(os.path.join(base_path, 'CITYSCAPES_RGB_input'))))):
        # if not (iter_l in shortlist_im_num):
        #     continue
        data[im_name] = {}
        list_of_images.append(im_name)
        data[im_name]['rgb'] = resize_anyto1024x512(os.path.join(base_path, 'CITYSCAPES_RGB_input', im_name))
        data[im_name]['segm_gt_color'] = resize_anyto1024x512(
            os.path.join(base_path, 'CITYSCAPES_segm_gt', im_name.replace('_leftImg8bit.png', '_gtFine_color.png')))
        data[im_name]['segm_gt'] = colormaps2label(data[im_name]['segm_gt_color'])
        data[im_name]['segm_gt_edges'], data[im_name]['segm_gt_non_edges'] = compute_segm_edges(
            data[im_name]['segm_gt'])
        data[im_name]['segm_gt_edges_color'], data[im_name]['segm_gt_non_edges_color'] = \
            label2colormap(data[im_name]['segm_gt_edges'][:, :, 0].numpy()), \
            label2colormap(data[im_name]['segm_gt_non_edges'][:, :, 0].numpy())
        for method in methods:
            if method in ['Source_Only', 'BDL', 'ProDA', 'CDBP_553', 'CDBP_576']:
                im_path = os.path.join(base_path, method + '_segm_output',
                                       im_name.replace('_leftImg8bit.png', '_leftImg8bit_color.png'))
            else:
                im_path = os.path.join(base_path, method + '_segm_output', im_name)
            data[im_name][method + '_color'] = resize_anyto1024x512(im_path)
            data[im_name][method] = colormaps2label(resize_anyto1024x512(im_path))

            cty_running_metrics[method]['whole'].update(data[im_name]['segm_gt'].numpy(),
                                                        data[im_name][method].numpy())
            cty_running_metrics[method]['edges'].update(data[im_name]['segm_gt_edges'][:, :, 0].numpy(),
                                                        data[im_name][method].numpy())
            cty_running_metrics[method]['non_edges'].update(data[im_name]['segm_gt_non_edges'][:, :, 0].numpy(),
                                                            data[im_name][method].numpy())

    return data, cty_running_metrics, list_of_images


def figs_paper(base_path):
    save_folder_path = base_path
    shortlist_im_num = [5, 17, 43, 130, 211, 324, 352, 378, 390, 440]
    # shortlisted_im_names = []
    # for iter_l in shortlist_im_num:
    #    shortlisted_im_names.append(list_of_images[iter_l])
    data, cty_running_metrics, shortlisted_im_names = load_images_folder(base_path, shortlist_im_num)

    print('Total number of images: {}'.format(len(data)))
    print_miou(cty_running_metrics)
    # for k, im in data[shortlisted_im_names[0]].items():
    #     print('key: {}, min: {}, max: {}, shape: {}'.format(k, im.min(), im.max(), im.shape))

    # Create final figure for the paper
    rows_zeros = np.ones((20, 6244, 3)) * 255
    cols_zeros = np.ones((512, 20, 3)) * 255
    im_cat_all = []

    # diff = np.array(data_mIoU['TridentAdapt_mIoU']) - np.array(data_mIoU['TIR_mIoU'])
    # shortlist_im_num = np.where(diff < -0.1)[0]
    # shortlist_im_num =
    # [(9, 6), (19, 8), (49, 4), (139, 1), (219, 2), (329, 5), (359, 3), (379, 9), (399, 1), (449, 1)]
    for iter_l, im_name in enumerate(shortlisted_im_names):
        zeros_ = torch.zeros_like(data[im_name]['rgb'])
        ones_ = torch.ones_like(data[im_name]['rgb']) * 255
        im_cat = []
        im_cat_up = [data[im_name]['rgb'], cols_zeros, data[im_name]['segm_gt_color']]
        im_cat_down = [ones_, cols_zeros, zeros_]
        im_cat_edges = [data[im_name]['rgb'], cols_zeros, data[im_name]['segm_gt_edges_color']]
        im_cat_non_edges = [data[im_name]['rgb'], cols_zeros, data[im_name]['segm_gt_non_edges_color']]
        for method in methods:
            im_cat_up.append(cols_zeros)
            im_cat_down.append(cols_zeros)
            im_cat_edges.append(cols_zeros)
            im_cat_non_edges.append(cols_zeros)

            # Semantic edges error map --> red - wrong prediction & green - correct prediction
            error_map = data[im_name]['segm_gt'] == data[im_name][method]
            error_map_color = torch.zeros_like(data[im_name]['rgb'])
            error_map_color = data[im_name][method + '_color'].clone()
            for i in range(3):
                error_map_color[:, :, i] *= data[im_name]['segm_gt'] != -1
                error_map_color[error_map] = 0

            # extract the edges on top of segmentation prediction
            pred_edge_map, pred_non_edge_map = compute_segm_edges(
                data[im_name][method].clone().float())  # * (data[im_name]['segm_gt_edges'] != -1).float()

            im_cat_up.append(data[im_name][method + '_color'])
            im_cat_down.append(error_map_color)
        im_cat.append(np.concatenate((np.concatenate(im_cat_up, 1), rows_zeros), 0))
        # im_cat_edges.append(data[im_name][method])
        im_cat = np.concatenate(im_cat, 1)

        im_cat_all.append(im_cat)
        im_cat_all.append(rows_zeros)
        if (iter_l + 1) % 10 == 0:
            im_cat_all = np.concatenate(im_cat_all, 0)
            Image.fromarray(np.array(im_cat_all, dtype=np.uint8)).save(
                os.path.join(save_folder_path, 'im_cat_all/supp_' + str(iter_l) + '.pdf'))
            im_cat_all = []

    im_cat_all = np.concatenate(im_cat_all, 0)
    plt.imshow(im_cat_all / 255)
    half_size = int(im_cat_all.shape[0] / 2)
    im_cat_1 = im_cat_all[:half_size]
    im_cat_2 = im_cat_all[half_size:]
    Image.fromarray(np.uint8(im_cat_1)).save(os.path.join(save_folder_path + 'im_cat/suppl_seg_1.png'))
    Image.fromarray(np.uint8(im_cat_1)).save(os.path.join(save_folder_path + 'im_cat/suppl_seg_1.pdf'))
    Image.fromarray(np.uint8(im_cat_2)).save(os.path.join(save_folder_path + 'im_cat/suppl_seg_2.png'))
    Image.fromarray(np.uint8(im_cat_2)).save(os.path.join(save_folder_path + 'im_cat/suppl_seg_2.pdf'))

    Image.fromarray(np.array(im_cat_all, dtype=np.uint8)). \
        save(os.path.join(save_folder_path, 'suppl_fig_' + str(iter_l) + 'woso.png'), quality=100)

    TheEnd = 1


def load_rgb_image(file_path, crop_size=None,
                   mean=np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)):
    if crop_size is None:
        crop_size = [512, 1024]

    image = Image.open(file_path).convert('RGB')

    # resize
    image = np.asarray(image.resize((crop_size[1], crop_size[0]), Image.BICUBIC), np.float32)
    image_np = image.copy().astype(np.uint8)
    image = image[:, :, ::-1] - mean  # change to BGR and normalize
    image = torch.tensor(image.transpose((2, 0, 1)) / 128.0).unsqueeze(0)

    return image, image_np


def video_suppl_material(image_folder_path=None, version='frankfurt'):
    if image_folder_path is None:
        image_folder_path = '/mnt/ssd1/Datasets/CITYSCAPES_/leftImg8bit/demoVideo/'

    # TridentAdapt network related stuff
    model_dict = {}
    device = torch.device('cuda:2')
    enc_shared = SharedEncoder().to(device)
    model_dict['enc_shared'] = enc_shared
    load_models(model_dict, '/mnt/ssd5/sharing-is-caring/TridentAdapt_TheBest_Weights')
    enc_shared.eval()
    upsample_512 = nn.Upsample(size=[512, 1024], mode='bilinear', align_corners=True)

    # Image reading, loading
    im_path_list = []
    for base_path_l, base_folder, file_paths in sorted(os.walk(image_folder_path)):
        for file_path in sorted(file_paths):
            if file_path.endswith('.png') or file_path.endswith('.jpg'):
                im_path_list.append(os.path.join(base_path_l, file_path))
    print('Total number of images in {} dataset: {}'.format('cityscapes-demo-video', len(im_path_list)))

    # Create demo video
    save_folder_path = '/mnt/ssd5/HDD_akhil/Pytorch/TridentAdapt/Documents/'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_rgb_segm_cat = cv2.VideoWriter(os.path.join(save_folder_path, 'videos', 'demo_' + version + '_cat.avi'),
                                         fourcc, 20.0, (640, 482))
    video_rgb_segm_mix = cv2.VideoWriter(os.path.join(save_folder_path, 'videos', 'demo_' + version + '_mix.avi'),
                                         fourcc, 20.0, (1024, 512))

    for iter_l in tqdm(range(len(im_path_list))):
        file_path = im_path_list[iter_l]
        rgb_im, rgb_np = load_rgb_image(file_path)
        images_val = Variable(rgb_im.to(device), requires_grad=False)

        # TridentAdapt model
        _, _, segm_prob, _ = enc_shared(images_val)
        segm_prob = upsample_512(segm_prob).data.cpu().numpy()[0].transpose(1, 2, 0)
        segm_im = np.asarray(np.argmax(segm_prob, axis=2), dtype=np.uint8)
        segm_color = np.array(colorize_mask(segm_im).convert('RGB'), dtype=np.float32)

        # RGB segm mix
        rgb_segm_color_np = np.uint8(0.35 * rgb_np + 0.65 * segm_color)
        rgb_segm_color_pil = Image.fromarray(rgb_segm_color_np)
        rgb_segm_color_pil.save(
            os.path.join(save_folder_path, 'TridentAdapt_segm_output_val_allFrames', file_path.split('/')[-1]))
        video_rgb_segm_mix.write(rgb_segm_color_np[:, :, ::-1])

    video_rgb_segm_mix.release()
    video_rgb_segm_cat.release()


def compute_segm_edges(segm_im):
    if type(segm_im) is not torch.tensor:
        segm_im = torch.tensor(segm_im).unsqueeze(0).float()

    edges = kornia.laplacian(segm_im.unsqueeze(0), kernel_size=5).squeeze(0)
    segm_edges = (torch.abs(edges[0]) > 0.1).long()

    # Apply dilation to improve the edge border region
    kernel = np.ones((5, 5), dtype=np.uint8)
    segm_edges = torch.tensor(cv2.dilate(segm_edges.float().numpy(), kernel=kernel, iterations=1))

    h, w = segm_edges.shape
    # segm_edges[:50, :] = 0
    # segm_edges[:, :50] = 0
    # segm_edges[h - 50:, :] = 0
    # segm_edges[:, w - 50:] = 0
    segm_edges_copy = segm_edges.clone()

    non_segm_edges = torch.abs(segm_edges - 1)

    # Expand 2D array to 3D array
    segm_edges = torch.cat((segm_edges.unsqueeze(-1), segm_edges.unsqueeze(-1), segm_edges.unsqueeze(-1)), 2)
    segm_edges[segm_edges == 0] = -1
    # extract semantic class information at the edges
    segm_gt_edges = segm_edges * (segm_im.permute(1, 2, 0) + 1)
    segm_gt_edges[segm_gt_edges <= 0] = 0
    segm_gt_edges -= 1

    non_segm_edges = torch.cat((non_segm_edges.unsqueeze(-1), non_segm_edges.unsqueeze(-1),
                                non_segm_edges.unsqueeze(-1)), 2)
    non_segm_edges[non_segm_edges == 0] = -1
    non_segm_edges = non_segm_edges * (segm_im.permute(1, 2, 0) + 1)
    non_segm_edges[non_segm_edges <= 0] = 0
    non_segm_edges -= 1

    return segm_gt_edges, non_segm_edges


valid_colors = [[128, 64, 128],
                [244, 35, 232],
                [70, 70, 70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170, 30],
                [220, 220, 0],
                [107, 142, 35],
                [152, 251, 152],
                [70, 130, 180],
                [220, 20, 60],
                [255, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                [0, 80, 100],
                [0, 0, 230],
                [119, 11, 32]]
label_colours = dict(zip(range(19), valid_colors))
n_classes = 19


def label2colormap(img):
    map = np.zeros((img.shape[0], img.shape[1], 3))
    temp = img[:, :]
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    map[:, :, :] = rgb
    map[map < 0] = 0
    return map


def colormaps2label(img):
    label = torch.zeros((img.shape[0], img.shape[1])) - 1
    for key, val in label_colours.items():
        r, g, b = val
        label[np.logical_and(np.logical_and(img[:, :, 0] == r, img[:, :, 1] == g), img[:, :, 2] == b)] = key
    return label.long()


def print_miou(cty_running_metrics):
    # Print mIoU results
    for method in methods:
        cty_score_whole, cty_class_iou_whole = cty_running_metrics[method]['whole'].get_scores()
        cty_score_edges, cty_class_iou_edges = cty_running_metrics[method]['edges'].get_scores()
        cty_score_non_edges, cty_class_iou_non_edges = cty_running_metrics[method]['non_edges'].get_scores()
        print('Method: {} '
              '\n Around semantic whole mIoU: {} '
              '\n Around semantic edges mIoU: {} '
              '\n Around Non semantic edges mIoU: {}'.
              format(method, cty_score_whole['Mean IoU : \t'], cty_score_edges['Mean IoU : \t'],
                     cty_score_non_edges['Mean IoU : \t']))

    TheEnd = 1


def plot_chart():
    plt.clf()
    font_size = 30
    data = {'exp_no': [1, 2, 3, 4, 5], 'val': [49.8, 52.7, 53.7, 54.5, 55.3]}
    plt.scatter(data['exp_no'][:-1], data['val'][:-1], s=400, c='blue')
    plt.scatter(data['exp_no'][:-1], data['val'][:-1], s=400, c='green')
    # plt.title('Error plot between 1.5 to 2.5 meters range')
    plt.xlabel('Experiment number', fontsize=font_size)
    plt.ylabel('mIoU (%)', fontsize=font_size)
    plt.grid(linestyle='--', linewidth=1)
    plt.xticks(fontsize=font_size)  # , color=[0.5, 0.5, 0.5])
    plt.yticks(fontsize=font_size)  # , color=[0.5, 0.5, 0.5])
    plt.axis([0, 6, 47.5, 57.5])


def bar_chart():
    # GTA5-to-Cityscapes adaptation results
    y_axis = [i for i in range(0, 102, 20)]
    legends = ['road', 'sdwk', 'bldng', 'wall', 'fence', 'pole', 'light', 'sign', 'veg', 'trrn', 'sky', 'psn', 'rider',
               'car', 'truck', 'bus', 'train', 'moto', 'bike', 'mIoU']
    source_only = [75.8, 16.8, 77.2, 12.5, 21.0, 25.5, 30.1, 20.1, 81.3, 24.6, 70.3, 53.8, 26.4, 49.9, 17.2, 25.9, 6.5,
                   25.3, 36.0, 36.6]  # 36.6
    CDBP_warm = [90.8, 40.6, 85.4, 40.0, 28.0, 34.5, 36.0, 19.9, 84.3, 31.2, 83.3, 63.5, 33.9, 88.8, 48.7, 49.9, 08.7,
                 27.0, 27.0, 48.5]  # 48.5
    CDBP_ours = [93.0, 54.4, 87.5, 47.0, 36.2, 41.6, 48.3, 40.1, 85.4, 36.2, 85.6, 65.9, 36.0, 90.2, 61.9, 57.7, 10.5,
                 33.7, 39.9, 55.3]  # 55.3

    # set width of bar
    barWidth, font_size = 0.25, 15
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Set position of bar on X axis
    br2 = np.arange(len(legends))
    br1 = [x - barWidth for x in br2]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    ax1.bar(br1, source_only, color=(1, 0.2, 0.2), width=barWidth, edgecolor='grey', label='Source Only')
    ax1.bar(br2, CDBP_warm, color=(0, 0.8, 0), width=barWidth, edgecolor='grey', label='CDBP (warm-up)')
    ax1.bar(br3, CDBP_ours, color=(0.2, 0.2, 1), width=barWidth, edgecolor='grey', label='CDBP (stage-1)')

    # Adding Xticks
    ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=font_size)
    ax1.set_xticks([i for i in range(len(legends))])
    ax1.set_xticklabels(legends, fontsize=font_size, rotation=45)
    ax1.set_yticks(y_axis)
    ax1.set_yticklabels(y_axis, fontsize=font_size)
    ax1.set_title("GTA5-to-Cityscapes", fontweight='bold', fontsize=20)

    ax1.legend(fontsize=font_size)
    ax1.grid(linestyle='--', linewidth=1)

    # Synthia-to-Cityscapes adaptation results.
    legends = ['road', 'sdwk', 'bldng', 'wall*', 'fence*', 'pole*', 'light', 'sign', 'veg', 'sky', 'psn', 'rider',
               'car', 'bus', 'mcycl', 'bcycl', 'mIoU', 'mIoU*']
    source_only = [55.6, 23.8, 74.6, 0, 0, 0, 6.1, 12.1, 74.8, 79.0, 55.3, 19.1, 39.6, 23.3, 13.7, 25.0, 0, 38.6]
    CDBP_warm = [87.0, 45.5, 81.7, 12.7, 0.6, 32.9, 14.8, 12.5, 82.6, 86.1, 56.3, 25.5, 85.7, 50.0, 25.3, 37.8, 46.1,
                 53.2]
    CDBP_ours = [89.2, 50.7, 84.6, 11.6, 3.0, 40.6, 35.0, 28.2, 84.8, 81.8, 61.1, 27.7, 88.4, 63.7, 30.9, 54.1, 52.2,
                 60.0]

    # Set position of bar on X axis
    br2 = np.arange(len(legends))
    br1 = [x - barWidth for x in br2]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    ax2.bar(br1, source_only, color=(1, 0.2, 0.2), width=barWidth, edgecolor='grey', label='Source Only')
    ax2.bar(br2, CDBP_warm, color=(0, 0.8, 0), width=barWidth, edgecolor='grey', label='CDBP (warm-up)')
    ax2.bar(br3, CDBP_ours, color=(0.2, 0.2, 1), width=barWidth, edgecolor='grey', label='CDBP (stage-1)')

    # Adding Xticks
    ax2.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=font_size)
    ax2.set_xticks([i for i in range(len(legends))])
    ax2.set_xticklabels(legends, fontsize=font_size, rotation=45)
    ax2.set_yticks(y_axis)
    ax2.set_yticklabels(y_axis, fontsize=font_size)
    ax2.set_title("Synthia-to-Cityscapes", fontweight='bold', fontsize=20)

    ax2.legend(fontsize=font_size)
    ax2.grid(linestyle='--', linewidth=1)


if __name__ == "__main__":
    base_path_g = '/mnt/ssd5/CDBP/qualitative_comparison/'
    # create semantic edges
    # segm_results_edges(base_path=base_path_g)

    figs_paper(base_path=base_path_g)

    # video_suppl_material(image_folder_path='/mnt/ssd1/Datasets/CITYSCAPES_/leftImg8bit_allFrames/val/frankfurt/')
