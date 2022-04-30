CUDAID=${1:-0}
SETTING=$2
REPEATTIME=${3:-5}
IMGFEAT=${4:-imagenet}
MODEL=imagenet_agent_bt
NAME=${IMGFEAT}_${SETTING}

flag="--attn soft 
      --train validlistener
      --load snap/$MODEL/state_dict/best_val_unseen
      --name $NAME
      --setting $SETTING
      --repeat_time $REPEATTIME  
      --features $IMGFEAT 
      --featdropout 0.3
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 80000 --maxAction 35"

CUDA_VISIBLE_DEVICES=$CUDAID python r2r_src/train.py $flag
