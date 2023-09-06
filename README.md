# Gradient-Guided Adaptive Domain Generalization for Cross Modality MRI Segmentation
## Guidelines
### 1. Install dependencies
We highly recommend you to create a new virtual environment for this project. The following command will install all the dependencies.
```bash
pip install -r requirements.txt
```
### 2. Download the dataset
We do not directly provide the dataset. You can download the datasets used in our experiments with instructions in the following links:
- [BraTS 2018](https://www.med.upenn.edu/sbia/brats2018/data.html)
- [MS-CMRSeg 2019](https://zmiclab.github.io/zxh/0/mscmrseg19/)

Once you download the datasets, please place the folders into `datasets` with the name of `BraTS2018_Raw` and `MS-CMRSeg2019_Raw` respectively. The folder structure should be like:
```
BraTS2018_Raw
├── HGG
│   ├── Brats18_2013_10_1
│   ├── Brats18_2013_11_1
│   ├── ...
│   └── Brats18_TCIA08_469_1
└── LGG
    ├── Brats18_2013_0_1
    ├── Brats18_2013_15_1
    ├── ...
    └── Brats18_TCIA13_654_1
```
```
MS-CMRSeg2019_Raw
├── C0LET2_gt_for_challenge19
│   ├── C0_manual_10
│   ├── LGE_manual_35_TestData
│   └── T2_manual_10
└── C0LET2_nii45_for_challenge19
    ├── c0gt
    ├── c0t2lge
    ├── lgegt
    └── t2gt
```
### 3. Preprocess the dataset
We provide the preprocessing script for each dataset. You can preprocess the dataset with the following commands:
#### BraTS2018
```bash
python datasets/BraTS_2018.py --root datasets/BraTS2018_Raw --save_dir datasets/BraTS --source <source_modality> --target <target_modality> --train_source True --val_target True
```
#### MS-CMRSeg2019
```bash
python datasets/MS-CMRSeg2019.py --root datasets/MS-CMRSeg2019_Raw --save_dir datasets/MS-CMRSeg2019 --source <source_modality> --target <target_modality> --train_source True --val_target True
```
### 4. Train the model
You can train the model with the following command:
```bash

```