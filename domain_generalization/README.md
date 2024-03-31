# DiGA: Distil to Generalize and then Adapt for Domain Adaptive Semantic Segmentation
**[[CVPR23 Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Shen_DiGA_Distil_To_Generalize_and_Then_Adapt_for_Domain_Adaptive_CVPR_2023_paper.pdf)**

## Datasets
The data folder `DiGA/domain_generalization/data` follows data structures below:
  ```
  |---data
      |--- Cityscapes
      |   |--- gtFine
      |   |--- leftImg8bit
      |--- GTA5
          |--- images
          |--- labels
      |--- BDD100k
          |--- images
               |--- val
          |--- gtFine
               |--- val
      |--- Mapillary
          |--- validation 
              |--- images
              |--- labels
  ```
## Usage

### Run DiGA Training Script Domain Generalization
#### Distillation Training ( e.g., GTA5 as source domain)
To start DiGA training for domain generalization, please run the following script,

```shell
python train_DiGA_DG.py --gta5_data_path --gta5_data_path ./data/GTA5 --city_data_path ./data/Cityscapes \
                                         --weight_dir ./work_dir/weights_DiGA_DG/      \
                                         --log_dir ./work_dir/log_DiGA_DG                 \
                                         --gen_img_dir ./work_dir/generated_imgs_DiGA_DG    \
                                         --cuda_device_id 0
```
Cityscapes data are loaded just for visualization and validation.

### Run Evaluation Script:
To evaluate the trained model on Cityscapes val set, run the following script,

```shell
python evaluate_val.py --cuda_device_id 0 --dataset_name 'Cityscapes' --data_path ./data/Cityscapes \
                       --data_list_path_val_img ./util/loader/cityscapes_list/val.txt \
                       --data_list_path_val_lbl ./util/loader/cityscapes_list/val_label.txt
```

To evaluate the trained model on BDD100k val set, run the following script,

```shell
python evaluate_val.py --cuda_device_id 0 --dataset_name 'BDD100k' --data_path ./data/BDD100k \
                       --data_list_path_val_img ./util/loader/bdd100k_list/val.txt  \
                       --data_list_path_val_lbl ./util/loader/bdd100k_list/val_label.txt
```

To evaluate the trained model on Mapillary val set, run the following script,

```shell
python evaluate_val.py --cuda_device_id 0 --dataset_name 'Mapillary' --data_path ./data/Mapillary \
                       --data_list_path_val_img ./util/loader/mapillary_list/val.txt \
                       --data_list_path_val_lbl ./util/loader/mapillary_list/val_label.txt
```

