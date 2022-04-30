CUDAID=${1:-0}
SETTING=$2
REPEATTIME=${3:-5}
IMGFEAT=${4:-imagenet}
MODEL=smna
NAME=${IMGFEAT}_${SETTING}

flag="--job cache
      --load_follower experiments/$MODEL/snapshots/follower_cg_pm_sample2step_imagenet_mean_pooled_1heads_train_iter_1900_val_unseen-success_rate=0.478 
      --experiment_name $NAME
      --setting $SETTING
      --repeat_time $REPEATTIME  
      --features $IMGFEAT 
      --max_episode_len 40 --K 20 --logit --beam"

CUDA_VISIBLE_DEVICES=$CUDAID python test_with_reranker.py $flag 
