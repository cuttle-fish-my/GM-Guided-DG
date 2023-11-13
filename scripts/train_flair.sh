TRAIN_FLAGS="--data_dir ../datasets/BraTS_flair2t1/train
            --use_fp16 True
            --save_dir ../saved_models/BraTS/flair/GMGDG
            --lr 1e-4
            --batch_size 24
            --save_interval 1000
            --lr_anneal_steps 10000
            --modality source
            --input_mode magnitude
            --in_channels 3
            --heavy_aug True
            --norm_type BN"

# activate your virtual environment (maybe?)
# source [your pip venv]
# or
# conda activate [your conda venv]

python ../src/Unet_train.py $TRAIN_FLAGS