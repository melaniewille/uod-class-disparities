# Class-Specific Performance Disparities in Underwater Object Detection
This is the official repository for the paper: "Are All Marine Species Created Equal? Performance Disparities in Underwater Object Detection". 

The full paper can be accessed on [arXiv](https://arxiv.org/abs/2508.18729) and a corresponding video on [YouTube](https://youtu.be/Gat4K3biW3I).

Our approach contributes to the field of underwater object detection by providing a systematic analysis that decomposes the object detection process into localization and classification stages to investigate underlying reasons behind class-specific performance differences. While previous works focus mainly on quantitative factors, we additionally consider the influence of intrinsic visual features on performance and prove that they are crucial to a species' detectability. The primary bottleneck for visually less distinctive targets lies in foreground-background distinction and they rely on negative examples from other classes. Therefore, the class distribution affects under-performing species in particular and needs to be chosen appropriately based on application needs. By offering these insights, our work supports advancements in underwater object detection to enable more efficient and reliable environmental monitoring, as well as sustainable fisheries and resource management.

This repository provides the steps and code to replicate our analysis on the DUO and RUOD-4C datasets, including the training of YOLO-11 and SSD detection models for localization, and ResNet-18, MobileNetV2 and VisionTransformer as classifiers.

If this repository and / or our paper contributes to your research, please consider citing the publication below.

### Bibtex
```
@article{wille2025all,
  title={Are All Marine Species Created Equal? Performance Disparities in Underwater Object Detection},
  author={Wille, Melanie and Fischer, Tobias and Raine, Scarlett},
  journal={arXiv preprint arXiv:2508.18729},
  year={2025}
}
```

## Table of Contents

- [Installation](#installation)  
- [Datasets](#datasets)  
- [Models](#models)  
- [Quick Start](#quick-start)  
- [Acknowledgements](#acknowledgements)

## Installation
We use [Pixi](https://pixi.org/) for environment and experiment management.  

If you do not have Pixi installed yet, you will need to do this once:

**On Linux / macOS**:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```
You might have to open a new terminal window after installation so the pixi command becomes available. You can verify the installation by running:
```bash
pixi --version
```
If it prints a version number, you’re ready to continue.

**On Windows (PowerShell)**:
```powershell
iwr -useb https://pixi.sh/install.ps1 | iex
```

**Next**:

Once completed, clone this repository and navigate into the directory:
```bash
git clone https://github.com/melaniewille/uod-class-disparities.git
cd uod-class-disparities
```

Then create the environment using:
```bash
cd pixi
pixi install
```
Pixi will automatically read the pixi.toml file and install all required dependencies.


## Datasets
Our datasets will be available on [Hugging Face](https://huggingface.co/).  

Please download:

- [**the localization datasets for DUO**](https://huggingface.co/datasets/melaniewille/uod-class-disparities-localization-study-DUO)
- [**the classification datasets for DUO**](https://huggingface.co/datasets/melaniewille/uod-class-disparities-classification-study-DUO)
- [**the localization datasets for RUOD-4C**](https://huggingface.co/datasets/melaniewille/uod-class-disparities-localization-study-RUOD-4C)
- [**the classification datasets for RUOD-4C**](https://huggingface.co/datasets/melaniewille/uod-class-disparities-classification-study-RUOD-4C)

and place them in the following folders within the repository:

- [experimental-studies-DUO/localization_study/loc_datasets/](./experimental-studies-DUO/localization_study/loc_datasets/)
- [experimental-studies-DUO/classification_study/cls_datasets/](./experimental-studies-DUO/classification_study/cls_datasets/)
- [experimental-studies-RUOD-4C/localization_study/loc_datasets/](./experimental-studies-DUO/localization_study/loc_datasets/)
- [experimental-studies-RUOD-4C/classification_study/cls_datasets/](./experimental-studies-DUO/classification_study/cls_datasets/)

```
.
├── README.md
├── experimental-studies-DUO/
│   ├── localization_study/
│   │   ├── loc_datasets/
│   │   └── ...
│   └── classification_study/
│       ├── cls_datasets/
│       └── ...
├── experimental-studies-RUOD-4C/
│   ├── localization_study/
│   │   ├── loc_datasets/
│   │   └── ...
│   └── classification_study/
│       ├── cls_datasets/
│       └── ...
└── pixi/
    ├── pixi.toml
    └── ... 
```

Once the datasets are placed correctly, all experiments can be executed from the [pixi/](./pixi/) folder using the commands defined in [pixi.toml](./pixi/pixi.toml), as described in the [Quick Start](#quick-start) section.        

## Models
We employ **YOLO11** and **SSD** object detection models in our **localization** study. Their training and testing scripts can be found in [experimental-studies-DUO/localization_study/](./experimental-studies-DUO/localization_study/)_architecture_ or [experimental-studies-RUOD-4C/localization_study/](./experimental-studies-RUOD-4C/localization_study/)_architecture_ for single-class, balanced and reduced dataset variations.

We further analyze **classification** performance using **ResNet-18**, **MobileNetV2** and **VisionTransformer**. Their scripts can be found in [experimental-studies-DUO/classification_study/](./experimental-studies-DUO/classification_study/)_architecture_ or [experimental-studies-RUOD-4C/classification_study/](./experimental-studies-RUOD-4C/classification_study/)_architecture_ for imbalanced, balanced and reduced data as well as for weighted sampling.

```
.
├── localization_study/
│   ├── YOLO11/
│   │   ├── YOLO11_single_train.py
│   │   ├── YOLO11_single_test.py
│   │   ├── YOLO11_balanced_train.py
│   │   ├── YOLO11_balanced_test.py
│   │   ├── YOLO11_reduced_train.py
│   │   └── YOLO11_reduced_test.py
│   └── SSD/
│   │   ├── SSD_single_class.py
│   │   ├── SSD_balanced.py
│   │   └── SSD_reduced.py
│   └── ...
└── classification_study/
    ├── ResNet18/
    │   ├── rn18_imbalanced_data_model.py
    │   ├── rn18_balanced_data_model.py
    │   ├── rn18_reduced_class_model.py
    │   └── rn18_weighted_sampling_model.py
    ├── MobileNetV2/
    │   ├── rn18_imbalanced_data_model.py
    │   ├── rn18_balanced_data_model.py
    │   ├── rn18_reduced_class_model.py
    │   └── rn18_weighted_sampling_model.py
    ├── VisionTransformer/
    │   ├── ViT_imbalanced_data_model.py
    │   ├── ViT_balanced_data_model.py
    │   ├── ViT_reduced_class_model.py
    │   └── ViT_weighted_sampling_model.py
    └── ...
```

The **results** reported in our **main paper** are based on the experiments conducted with **YOLO11** and **ResNet18**.
The clssification results from ResNet18 were taken as average across three independent runs of different seeds (45, 100, 550). When using this repository to repeat our analysis, the classification experiments executed through the pixi commands are set to the first run by default. For additional runs and seed variations, navigate to the .py script locations described above and manually change the run number in line 27 ad the seed in line 31.


## Quick Start
Ensure you have cloned the repository, set up the environment and downloaded the datasets according to the [Installation](#installation) and [Datasets](#datasets) sections.
To run the experiments, navigate into the `pixi/` folder and run the desired tasks, using the task names specified in the [pixi.toml](./pixi/pixi.toml):

```bash
cd pixi

# run on CPU
pixi run [task-name] 

# or run on GPU
pixi run -e cuda [task-name]
```
for **all localization experiments on DUO**:
```bash
pixi run loc-DUO-yolo-all
pixi run loc-DUO-ssd-all
```
or any of the following for the **individual localization experiments on DUO**:
```bash
pixi run loc-DUO-yolo-single-class
pixi run loc-DUO-yolo-balanced
pixi run loc-DUO-yolo-reduced
```
for **all localization experiments on RUOD-4C**:
```bash
pixi run loc-RUOD-4C-yolo-all
pixi run loc-RUOD-4C-ssd-all
```
or any of the following for the **individual localization experiments on RUOD-4C**:
```bash
pixi run loc-RUOD-4C-yolo-single-class
pixi run loc-RUOD-4C-yolo-balanced
pixi run loc-RUOD-4C-yolo-reduced
```
for **all classification experiments on DUO**:
```bash
pixi run cls-DUO-rn18-all
pixi run cls-DUO-mnv2-all
pixi run cls-DUO-ViT-all
```
or any of the following for the **individual classification experiments on DUO**:
```bash
pixi run cls-DUO-rn18-imbalanced
pixi run cls-DUO-rn18-balanced
pixi run cls-DUO-rn18-reduced
pixi run cls-DUO-rn18-weighted
pixi run cls-DUO-mnv2-imbalanced
pixi run cls-DUO-mnv2-balanced
pixi run cls-DUO-mnv2-reduced
pixi run cls-DUO-mnv2-weighted
pixi run cls-DUO-ViT-imbalanced
pixi run cls-DUO-ViT-balanced
pixi run cls-DUO-ViT-reduced
pixi run cls-DUO-ViT-weighted
```
for **all classification experiments on RUOD-4C**:
```bash
pixi run cls-RUOD-4C-rn18-all
pixi run cls-RUOD-4C-mnv2-all
pixi run cls-RUOD-4C-ViT-all
```
or any of the following for the **individual classification experiments on RUOD-4C**:
```bash
pixi run cls-RUOD-4C-rn18-imbalanced
pixi run cls-RUOD-4C-rn18-balanced
pixi run cls-RUOD-4C-rn18-reduced
pixi run cls-RUOD-4C-rn18-weighted
pixi run cls-RUOD-4C-mnv2-imbalanced
pixi run cls-RUOD-4C-mnv2-balanced
pixi run cls-RUOD-4C-mnv2-reduced
pixi run cls-RUOD-4C-mnv2-weighted
pixi run cls-RUOD-4C-ViT-imbalanced
pixi run cls-RUOD-4C-ViT-balanced
pixi run cls-RUOD-4C-ViT-reduced
pixi run cls-RUOD-4C-ViT-weighted
```

The results can be found in automatically generated results folders for each dataset (DUO / RUOD-4C) and study (localization / classification) split into the different architectures:
```
experimental-studies-[dataset]/
├── localization_study/
│   ├── loc_results/
│   │   ├── YOLO11/
│   │   └── SSD/
│   └── ...
└── classification_study/
    ├── cls_results/
    │   ├── ResNet18/
    │   ├── MobileNetV2/
    │   └── VisionTransformer/
    └── ...
```

Additional evaluation of bounding boxes and TIDE errors for localization is available under [experimental-studies-DUO/localization_study/loc_evaluation_scripts](./experimental-studies-DUO/localization_study/loc_evaluation_scripts/) or [experimental-studies-RUOD-4C/localization_study/loc_evaluation_scripts](./experimental-studies-RUOD-4C/localization_study/loc_evaluation_scripts/).

## Acknowledgements
This work was conducted at the **QUT Centre for Robotics**.

We also thank the contributors of the **DUO** dataset - check out their [Github](https://github.com/chongweiliu/DUO) and paper:
```
@inproceedings{liu2021dataset,
  title={A dataset and benchmark of underwater object detection for robot picking},
  author={Liu, Chongwei and Li, Haojie and Wang, Shuchang and Zhu, Ming and Wang, Dong and Fan, Xin and Wang, Zhihui},
  booktitle={2021 IEEE international conference on multimedia \& expo workshops (ICMEW)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
```
and the contributors of the **RUOD** dataset - check out their [Github](https://github.com/dlut-dimt/RUOD) and paper:
```
@article{fu2023rethinking,
  title={Rethinking general underwater object detection: Datasets, challenges, and solutions},
  author={Fu, Chenping and Liu, Risheng and Fan, Xin and Chen, Puyang and Fu, Hao and Yuan, Wanqi and Zhu, Ming and Luo, Zhongxuan},
  journal={Neurocomputing},
  volume={517},
  pages={243--256},
  year={2023},
  publisher={Elsevier}
}
```

We redistribute our derived versions of these two datasets according to:

- a **private** permission for the **DUO** dataset from the original authors (email correspondence, October 2025). Further redistribution or commercial use may require additional consent from Liu et al.
- the **Apache License 2.0** for the **RUOD** dataset, under which it was originally released. We release our subset RUOD-4C under the same license.




