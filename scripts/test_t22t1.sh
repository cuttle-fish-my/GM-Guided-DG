TRAIN_FLAGS="--data_dir ../datasets/BraTS_t22t1/val
                        --input_mode magnitude
                        --modality target
                        --in_channels 3
                        --model_path ../saved_models/BraTS/t2/GMGDG/model010000.pt
                        --norm_type BN
                        --save_dir ../val_res/BraTS/t22t1"
# activate your virtual environment (maybe?)
# source [your pip venv]
# or
# conda activate [your conda venv]

python ../src/Unet_val.py $TRAIN_FLAGS
