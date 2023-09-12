# Gradient-Map-Guided Adaptive Domain Generalization for Cross Modality MRI Segmentation
## Guidelines
### 0. Platform Support
We only guarantee the correctness of the code on the following platform:
* Linux
* MacOS (with `MPS` acceleration)
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
declare -a SOURCE=("t2" "flair")
declare -a TARGET=("t1" "t1ce")
for source in ${SOURCE[@]}
do
  for target in ${TARGET[@]}
  do
    python datasets/BraTS_2018.py \
            --root datasets/BraTS2018_Raw \
            --save_dir datasets/BraTS_2018 \
            --source $source \
            --target $target \
            --train_source True \
            --val_target True
  done
done
```
#### MS-CMRSeg2019
```bash
source="C0"
declare -a TARGET=("T2" "LGE")
for target in ${TARGET[@]}
do
  python datasets/MSCMRSeg2019.py \
          --root datasets/MS-CMRSeg2019_Raw \
          --save_dir datasets/MS-CMRSeg2019 \
          --source $source \
          --target $target \
          --train_source True \
          --val_target True
done
```
### 4. Train the model
You can train the model with the following command:
```bash
python ./scripts/Unet_train.py \
        --data_dir ./datasets/<DATASET>/train \
        --use_fp16 False \
        --save_dir ./saved_models/<EXP_Name> \
        --lr 1e-4 \
        --batch_size 24 \
        --save_interval 1000 \
        --lr_anneal_steps 10000 \
        --modality source \
        --input_mode magnitude \
        --heavy_aug True
```
You can visualize the intermediate results with the following command:
```bash
tensorboard --logdir ./saved_models/<EXP_NAME>
```
#### Remark: 
* `<DATASET>` is the name of generated folders in [Section 3](#3-preprocess-the-dataset).
* Only set `--use_fp16` `True` when using NVIDIA GPU.
### 5. Test the model
You can test the model with the following command:
```bash
python ./scripts/Unet_val.py \
--data_dir ./datasets/<DATASET>/val \
--save_dir ./val_res/<EXP_NAME> \
--model_path ./saved_models/<EXP_NAME>/model010000.pt \
--dropout 0.0 \
--input_mode magnitude \
--modality target \
--TTA_mode PseudoLabel \
--TTA_lr 1e-2 \
--TTA_steps 2 \
--TTA_episodic True \
--TTA_alpha <TTA_ALPHA> \
--TTA_class_idx 1 \
--lambda_BN 0.4 \
--lambda_ent 1 \
--lambda_consistency 1
```
#### Remark:
* `<TTA_ALPHA>` is `0.5` for BraTS2018 and `0.9` for MS-CMRSeg2019

If you want to see the segmentation results and formal evaluation metrics, use the following command:
```bash
tensorboard --logdir ./val_res/<EXP_NAME>
```

