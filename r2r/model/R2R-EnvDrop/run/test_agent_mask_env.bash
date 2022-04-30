CUDAID=${1:-0}
IMGFEATMODE=${2:-foreground}
IMGFEAT=${3:-imagenet}
SETTING=mask_env
MODEL=imagenet_agent_bt
IMGFEATPATTERN=ResNet-152-${IMGFEAT}_%s_m%.2f_%d.tsv
NAME=${IMGFEAT}_${SETTING}

flag="--attn soft 
      --train validlistener
      --load snap/$MODEL/state_dict/best_val_unseen
      --name $NAME
      --setting $SETTING
      --repeat_time 1
      --reset_img_feat 1 
      --features $IMGFEAT
      --img_feat_mode $IMGFEATMODE
      --img_feat_pattern $IMGFEATPATTERN
      --featdropout 0.3
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 80000 --maxAction 35"

CUDA_VISIBLE_DEVICES=$CUDAID python r2r_src/train.py $flag
