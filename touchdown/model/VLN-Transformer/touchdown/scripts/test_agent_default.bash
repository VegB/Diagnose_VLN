CUDAID=${1:-0}
MODEL=${2:-vlntrans}
IMG_FEAT_DIR=./ # MODIFY THIS TO THE IMAGE FEATURE DIRECTORY 

flags="--test True 
       --dataset touchdown
       --img_feat_dir ${IMG_FEAT_DIR}
       --model $MODEL
       --resume_from experiments 
       --resume TC_best"
       
CUDA_VISIBLE_DEVICES=$1 python main.py $flags
