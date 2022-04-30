CUDAID=${1:-0}
IMGFEAT=${2:-imagenet}
MODEL=smna
RANKER=released/candidates_ranker_28D_0.6321
NAME=${IMGFEAT}_default

flag="--job cache
      --load_follower experiments/$MODEL/snapshots/follower_cg_pm_sample2step_imagenet_mean_pooled_1heads_train_iter_1900_val_unseen-success_rate=0.478 
      --load_reranker candidates/$RANKER 
      --experiment_name $NAME
      --features $IMGFEAT
      --max_episode_len 40 --K 20 --logit --beam"

CUDA_VISIBLE_DEVICES=$CUDAID python test_with_reranker.py $flag 
