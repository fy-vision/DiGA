# DiGA: Distil to Generalize and then Adapt for Domain Adaptive Semantic Segmentation
**[[CVPR23 Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Shen_DiGA_Distil_To_Generalize_and_Then_Adapt_for_Domain_Adaptive_CVPR_2023_paper.pdf)**

## Datasets
The data folder `DiGA/semi-supervised_segmentation/data` follows the original structure the Cityscapes dataset:
  ```
  |---data
      |--- Cityscapes
      |   |--- gtFine
      |   |--- leftImg8bit
  ```
## Usage

### Run Training Script Semi-Supervised Semantic Segmentation
We take the 1-16 split training as an example. (Download hrnetv2_w48_imagenet_pretrained.pth from [link](https://github.com/HRNet/HRNet-Image-Classification) and save to ./model/backbone_pretrained/)

#### Distillation Warm-up Stage 
To start DiGA training for the warm-up stage, please run the following script,

```shell
python train_DiGA_semiseg_warm_up.py  --city_data_path ./data/Cityscapes \
                                      --weight_dir ./work_dir/weights_DiGA_semiseg_1_16_warm_up/      \
                                      --log_dir ./work_dir/log_DiGA_semiseg_1_16_warm_up                 \
                                      --gen_img_dir ./work_dir/generated_imgs_DiGA_semiseg_1_16_warm_up     \
                                      --data_list_path_city_img_gt ./util/loader/cityscapes_list/split_train/train_image_labeled_1-16.txt \
                                      --data_list_path_city_lbl_gt /util/loader/cityscapes_list/split_train/train_label_labeled_1-16.txt \
                                      --data_list_path_city_img ./util/loader/cityscapes_list/split_train/train_image_unlabeled_1-16.txt \
                                      --data_list_path_city_lbl /util/loader/cityscapes_list/split_train/train_label_unlabeled_1-16.txt \
                                      --cuda_device_id 0
```

#### Self-training Stage
Before the self-training stage, generate pseudo-labels by running the following script, 

```shell
python pseudolabel_generator.py --city_data_path ./data/Cityscapes \
                                --weight_dir ./work_dir/weights_DiGA_semiseg_1_16_warm_up/  \
                                --output_dir ./data/Cityscapes/pseudo_train_1_16_warm_up \
                                --cuda_device_id 0                           
```

Afterwards, calculate the feature class centroids by running the following script,

```shell
python calc_centroids.py --city_data_path ./data/Cityscapes \
                         --weight_dir ./work_dir/weights_DiGA_semiseg_1_16_warm_up/  \
                         --centroid_dir ./work_dir/class_centroids_1_16/  \
                         --cuda_device_id 0                                     
```

Start the self-training stage by running the following script,

```shell
python train_DiGA_semiseg_self_training.py --city_data_path ./data/Cityscapes \
                                           --load_weight_dir ./work_dir/weights_DiGA_semiseg_1_16_warm_up/       \
                                           --save_weight_dir ./work_dir/weights_DiGA_semiseg_1_16_ST/       \
                                           --log_dir ./work_dir/log_DiGA_semiseg_1_16_ST                 \
                                           --gen_img_dir ./work_dir/generated_imgs_DiGA_semiseg_1_16_ST     \
                                           --pseudo_dir pseudo_train_1_16_warm_up \
                                           --centroid_dir ./work_dir/class_centroids_1_16/feat_centroids   \
                                           --data_list_path_city_img_gt ./util/loader/cityscapes_list/split_train/train_image_labeled_1-16.txt \
                                           --data_list_path_city_lbl_gt ./util/loader/cityscapes_list/split_train/train_label_labeled_1-16.txt \
                                           --data_list_path_city_img ./util/loader/cityscapes_list/split_train/train_image_unlabeled_1-16.txt \
                                           --data_list_path_city_lbl ./util/loader/cityscapes_list/split_train/train_label_unlabeled_1-16.txt \
                                           --cuda_device_id 0                               
```

If we do not want to overwrite the current model weights, please make a copy. Before the refined self-training stage starts, generate pseudo-labels by running the following script, 

```shell
python pseudolabel_generator.py --city_data_path ./data/Cityscapes \
                                --weight_dir ./work_dir/weights_DiGA_semiseg_1_16_warm_up/  \
                                --output_dir ./data/Cityscapes/pseudo_train_1_16_ST \
                                --cuda_device_id 0                           
```

then run the self-training script again,

```shell
python train_DiGA_semiseg_self_training.py --city_data_path ./data/Cityscapes  \
                                           --load_weight_dir ./work_dir/weights_DiGA_semiseg_1_16_ST/       \
                                           --save_weight_dir ./work_dir/weights_DiGA_semiseg_1_16_ST/       \
                                           --log_dir ./work_dir/log_DiGA_semiseg_1_16_ST                 \
                                           --gen_img_dir ./work_dir/generated_imgs_DiGA_semiseg_1_16_ST     \
                                           --pseudo_dir pseudo_train_1_16_ST \
                                           --centroid_dir ./work_dir/class_centroids_1_16/feat_centroids   \
                                           --data_list_path_city_img_gt ./util/loader/cityscapes_list/split_train/train_image_labeled_1-16.txt \
                                           --data_list_path_city_lbl_gt ./util/loader/cityscapes_list/split_train/train_label_labeled_1-16.txt \
                                           --data_list_path_city_img ./util/loader/cityscapes_list/split_train/train_image_unlabeled_1-16.txt \
                                           --data_list_path_city_lbl ./util/loader/cityscapes_list/split_train/train_label_unlabeled_1-16.txt \
                                           --cuda_device_id 0  
```

### Run Evaluation Script:
```shell
python evaluate_val.py --city_data_path ./data/Cityscapes --weight_dir PATH-TO-WEIGHT_DIR --cuda_device_id 0
```


