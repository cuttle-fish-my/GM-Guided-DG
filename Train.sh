declare -a SOURCE=("t2" "flair")
declare -a TARGET=("t1" "t1ce")
for source in ${SOURCE[@]}
do
  for target in ${TARGET[@]}
  do
    python datasets/BraTS_2018.py --root datasets/BraTS2018_Raw --save_dir datasets/BraTS --source $source --target $target --train_source True --val_target True
  done
done