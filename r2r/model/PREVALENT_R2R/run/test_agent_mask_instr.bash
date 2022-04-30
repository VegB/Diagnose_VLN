CUDAID=${1:-0}
SETTING=$2
REPEATTIME=${3:-5}
IMGFEAT=${4:-imagenet}
MODEL=cvpr_agent
NAME=${IMGFEAT}_${SETTING}

flag="--attn soft 
      --train validlistener 
      --load snap/$MODEL/state_dict/best_val_unseen
      --name $NAME
      --setting $SETTING
      --repeat_time $REPEATTIME  
      --features $IMGFEAT 
      --selfTrain 
      --aug tasks/R2R/data/aug_paths.json 
      --speaker snap/speaker/state_dict/best_val_unseen_bleu 
      --pretrain_model_name ./pretrained_hug_models/dicadd/checkpoint-12864 
      --angleFeatSize 128 
      --accumulateGrad 
      --featdropout 0.4 
      --feedback sample 
      --subout max 
      --optim rms 
      --lr 0.00002 
      --iters 100000 
      --maxAction 35 
      --encoderType Dic 
      --batchSize 20 
      --include_vision True 
      --use_dropout_vision True 
      --d_enc_hidden_size 1024 
      --critic_dim 1024"

CUDA_VISIBLE_DEVICES=$CUDAID python r2r_src/train.py $flag
