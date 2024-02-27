TRAIN_FLAGS="--data_dir ../datasets/MS-CMRSeg2019_C02LGE/val
                        --input_mode magnitude
                        --modality target
                        --in_channels 3
                        --model_path ../saved_models/MS-CMRSeg2019/C0/GMGDG/model010000.pt
                        --norm_type BN
                        --dropout 0.0
                        --save_dir ../val_res/MS-CMRSeg2019/C02LGE
                        --TTA_mode PseudoLabel
                        --TTA_lr 1e-2
                        --TTA_steps 2
                        --TTA_episodic True
                        --TTA_alpha 0.9
                        --TTA_class_idx 1
                        --lambda_BN 0.4
                        --lambda_ent 1
                        --lambda_consistency 1
                        "
# activate your virtual environment (maybe?)
# source [your pip venv]
# or
# conda activate [your conda venv]

python ../src/Unet_val.py $TRAIN_FLAGS
