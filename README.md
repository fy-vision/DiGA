# DiGA: Distil to Generalize and then Adapt for Domain Adaptive Semantic Segmentation
**[[CVPR23 Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Shen_DiGA_Distil_To_Generalize_and_Then_Adapt_for_Domain_Adaptive_CVPR_2023_paper.pdf)**
## Abstract
>Domain adaptive semantic segmentation methods commonly utilize stage-wise training, consisting of a warm-up and a self-training stage. However, this popular approach still faces several challenges in each stage: for warm-up, the widely adopted adversarial training often results in limited performance gain, due to blind feature alignment; for self-training, finding proper categorical thresholds is very tricky. To alleviate these issues, we first propose to replace the adversarial training in the warm-up stage by a novel symmetric knowledge distillation module that only accesses the source domain data and makes the model domain generalizable. Surprisingly, this domain generalizable warm-up model brings substantial performance improvement, which can be further amplified via our proposed cross-domain mixture data augmentation technique. Then, for the self-training stage, we propose a threshold-free dynamic pseudo-label selection mechanism to ease the aforementioned threshold problem and make the model better adapted to the target domain. Extensive experiments demonstrate that our framework achieves remarkable and consistent improvements compared to the prior arts on popular benchmarks.

## Setup Environment

The scripts run smoothly with python 3.9.4 and CUDA 11.3 on a NVIDIA RTX 8000. To run our scripts, we suggest setting up the following virtual environment:

```shell
python -m venv ~/venv/diga
source ~/venv/diga/bin/activate
```

The required python packages of this environment can be installed by the following command:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage
**Clone** this github repository:
```bash
  git clone https://github.com/fy-vision/DiGA
  cd DiGA
```

### Running DiGA Scripts for Unsupervised Domain Adaptation:
Please go to ./domain_adaptation folder

```shell
cd domain_adaptation
```

If you are looking for a specific adaptation benchmark e.g. Synthia->Cityscapes, please go to ./Synthia folder and refer to [domain adaptation/Synthia/README.md](domain_adaptation/Synthia/README.md) 

```shell
cd Synthia
```
### Running DiGA Scripts for Domain Generalization:
Please go to ./domain_generalization folder

```shell
cd domain_generalization
```

and refer to [domain_generalization/README.md](domain_generalization/README.md)

### Running DiGA Scripts for Semi-Supervised Semantic Segmentation:
Please go to ./semi-supervised segmentation folder

```shell
cd semi-supervised_segmentation
```

and refer to [semi-supervised_segmentation/README.md](semi-supervised_segmentation/README.md)


## Citation
If you like this work and would like to use our code or models for research, please consider citing our paper:
```
@inproceedings{shen2023diga,
  title={DiGA: Distil to generalize and then adapt for domain adaptive semantic segmentation},
  author={Shen, Fengyi and Gurram, Akhil and Liu, Ziyuan and Wang, He and Knoll, Alois},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15866--15877},
  year={2023}
}
```
## Acknowledgement
Our implementation is inspired by [ProDA](https://github.com/microsoft/ProDA) and [TridentAdapt](https://github.com/HMRC-AEL/TridentAdapt).
