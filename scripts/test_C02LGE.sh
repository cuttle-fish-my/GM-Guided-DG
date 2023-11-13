TRAIN_FLAGS="--data_dir ../datasets/MS-CMRSeg2019_C02LGE/val
                        --input_mode magnitude
                        --modality target
                        --in_channels 3
                        --model_path ../saved_models/MS-CMRSeg2019/C0/GMGDG/model010000.pt
                        --norm_type BN
                        --save_dir ../val_res/MS-CMRSeg2019/C02LGE"
# activate your virtual environment (maybe?)
# source [your pip venv]
# or
# conda activate [your conda venv]

python ../src/Unet_val.py $TRAIN_FLAGS
