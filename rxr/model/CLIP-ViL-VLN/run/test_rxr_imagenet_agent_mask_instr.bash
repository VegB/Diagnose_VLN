CUDAID=${1:-0}
SETTING=$2
REPEATTIME=${3:-5}
MODEL=rxr_imagenet/agent_rxr_en_maxinput160_ml04
IMGFEAT=imagenet
name=mask_${IMGFEAT}_${SETTING}

flag="--attn soft
      --train validlistener 
      --load snap/$MODEL/state_dict/best_val_unseen
      --name $name
      --setting $SETTING
      --repeat_time $REPEATTIME
      --featdropout 0.3
      --angleFeatSize 128
      --language en
      --maxInput 160
      --features $IMGFEAT
      --feature_size 2048
      --feedback sample
      --mlWeight 0.4
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 80000 --maxAction 35"

CUDA_VISIBLE_DEVICES=$CUDAID python rxr_src/train.py $flag 
