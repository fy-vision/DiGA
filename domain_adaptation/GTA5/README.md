# DiGA: Distil to Generalize and then Adapt for Domain Adaptive Semantic Segmentation
**[[CVPR23 Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Shen_DiGA_Distil_To_Generalize_and_Then_Adapt_for_Domain_Adaptive_CVPR_2023_paper.pdf)**

## Datasets
The data folder `DiGA/domain_adaptation/GTA5/data` follows the original structure of each dataset (e.g. GTA5-->Cityscapes):
  ```
  |---data
      |--- Cityscapes
      |   |--- gtFine
      |   |--- leftImg8bit
      |--- GTA5
          |--- images
          |--- labels
  ```
## Usage

### Run Training Script for GTA5->Cityscapes Adaptation
#### Image Domain Translator
To prepare the image domain translator, please run the following script,

```shell
python train_domain_translator.py --gta5_data_path ./data/GTA5 --city_data_path ./data/Cityscapes \
                                  --weight_dir ./work_dir/weights_DiGA_warm_up/      \
                                  --log_dir ./work_dir/log_domain_translator                  \
                                  --gen_img_dir ./work_dir/generated_imgs_domain_translator     \
                                  --cuda_device_id 0
                                  
```

#### Distillation Warm-up Stage 
To start DiGA training for the warm-up stage, please run the following script,

```shell
python train_DiGA_gta2city_warm_up.py --gta5_data_path ./data/GTA5 --city_data_path ./data/Cityscapes \
                                      --weight_dir ./work_dir/weights_DiGA_warm_up/      \
                                      --log_dir ./work_dir/log_DiGA_warm_up                 \
                                      --gen_img_dir ./work_dir/generated_imgs_DiGA_warm_up     \
                                      --cuda_device_id 0
```

#### Self-training Stage
Before the self-training stage, generate pseudo-labels by running the following script, 

```shell
python pseudolabel_generator.py --city_data_path ./data/Cityscapes \
                                --weight_dir ./work_dir/weights_DiGA_warm_up/  \
                                --output_dir ./data/Cityscapes/pseudo_train_warm_up \
                                --cuda_device_id 0                           
```

Afterwards, calculate the feature class centroids by running the following script,

```shell
python calc_centroids.py --gta5_data_path ./data/GTA5 --city_data_path ./data/Cityscapes \
                         --weight_dir ./work_dir/weights_DiGA_warm_up/  \
                         --centroid_dir ./work_dir/class_centroids/ \
                         --cuda_device_id 0                   
```
Start the self-training stage by running the following script,

```shell
python train_DiGA_gta2city_self_training.py --gta5_data_path ./data/GTA5 --city_data_path ./data/Cityscapes \
                                            --load_weight_dir ./work_dir/weights_DiGA_warm_up/      \
                                            --save_weight_dir ./work_dir/weights_DiGA_ST/      \
                                            --pseudo_dir pseudo_train_warm_up                \
                                            --centroid_dir ./work_dir/class_centroids/feat_centroids   \
                                            --log_dir ./work_dir/log_DiGA_ST                 \
                                            --gen_img_dir ./work_dir/generated_imgs_DiGA_ST     \
                                            --cuda_device_id 0                                     
```

If we do not want to overwrite the current model weights, please make a copy. Before the refined self-training stage starts, generate pseudo-labels by running the following script, 

```shell
python pseudolabel_generator.py --city_data_path ./data/Cityscapes \
                                --weight_dir ./work_dir/weights_DiGA_ST/  \
                                --output_dir ./data/Cityscapes/pseudo_train_ST \
                                --cuda_device_id 0                           
```

then run the self-training script again,

```shell
python train_DiGA_gta2city_self_training.py --gta5_data_path ./data/GTA5 --city_data_path ./data/Cityscapes  \
                                            --load_weight_dir ./work_dir/weights_DiGA_ST/      \
                                            --save_weight_dir ./work_dir/weights_DiGA_ST/      \
                                            --pseudo_dir pseudo_train_ST                \
                                            --centroid_dir ./work_dir/class_centroids/feat_centroids   \
                                            --log_dir ./work_dir/log_DiGA_ST                 \
                                            --gen_img_dir ./work_dir/generated_imgs_DiGA_ST     \
                                            --cuda_device_id 0   
```

### Run Evaluation Script:
```shell
python evaluate_val.py --city_data_path ./data/Cityscapes --weight_dir PATH-TO-WEIGHT_DIR --cuda_device_id 0
```


