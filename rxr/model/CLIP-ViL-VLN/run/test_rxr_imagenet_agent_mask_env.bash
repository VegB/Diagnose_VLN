CUDAID=${1:-0}
IMGFEATMODE=${2:-foreground}
SETTING=mask_env
MODEL=rxr_imagenet/agent_rxr_en_maxinput160_ml04
IMGFEATPATTERN=ResNet-152-imagenet_%s_m%.2f_%d.tsv
IMGFEAT=imagenet
name=mask_env

flag="--attn soft
      --train validlistener 
      --load snap/$MODEL/state_dict/best_val_unseen
      --name $name
      --setting $SETTING
      --repeat_time 1
      --featdropout 0.3
      --angleFeatSize 128
      --language en
      --maxInput 160
      --features $IMGFEAT
      --feature_size 2048
      --reset_img_feat 1
      --img_feat_pattern $IMGFEATPATTERN
      --img_feat_mode $IMGFEATMODE
      --feedback sample
      --mlWeight 0.4
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 80000 --maxAction 35"

CUDA_VISIBLE_DEVICES=$CUDAID python rxr_src/train.py $flag
