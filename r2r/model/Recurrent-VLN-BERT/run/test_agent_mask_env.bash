CUDAID=${1:-0}
IMGFEATMODE=${2:-foreground}
IMGFEAT=${3:-places365}
SETTING=mask_env
MODEL=VLNBERT-train-OriginalR2R
IMGFEATPATTERN=ResNet-152-${IMGFEAT}_%s_m%.2f_%d.tsv
NAME=${IMGFEAT}_${SETTING}

rate=$2
repeat=$3
name=test_VLNBERT_mask_env_${rate}_${repeat}

flag="--vlnbert oscar
      --submit 0
      --test_only 0
      --train validlistener
      --load snap/$MODEL/state_dict/best_val_unseen
      --name $NAME
      --setting $SETTING
      --repeat_time 1
      --reset_img_feat 1
      --features $IMGFEAT
      --img_feat_mode $IMGFEATMODE
      --img_feat_pattern $IMGFEATPATTERN
      --maxAction 15
      --batchSize 8
      --feedback sample
      --lr 1e-5
      --iters 300000
      --optim adamW
      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5"

CUDA_VISIBLE_DEVICES=$CUDAID python r2r_src/train.py $flag 
