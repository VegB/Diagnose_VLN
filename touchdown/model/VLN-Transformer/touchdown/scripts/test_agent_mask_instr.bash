CUDAID=${1:-0}
MODEL=${2:-vlntrans}
SETTING=$3
REPEATTIME=${4:-5}
IMG_FEAT_DIR=./ # MODIFY THIS TO THE IMAGE FEATURE DIRECTORY 

flags="--test True 
       --dataset touchdown
       --img_feat_dir ${IMG_FEAT_DIR}
       --model $MODEL
       --setting $SETTING
       --repeat_time $REPEATTIME  
       --resume_from experiments 
       --resume TC_best"
       
CUDA_VISIBLE_DEVICES=$1 python main.py $flags
