CUDAID=${1:-0}
IMGFEATMODE=${2:-foreground}
IMGFEAT=${3:-imagenet}
SETTING=mask_env
MODEL=smna
IMGFEATPATTERN=ResNet-152-${IMGFEAT}_%s_m%.2f_%d.tsv
NAME=${IMGFEAT}_${SETTING}

flag="--job cache
      --load_follower experiments/$MODEL/snapshots/follower_cg_pm_sample2step_imagenet_mean_pooled_1heads_train_iter_1900_val_unseen-success_rate=0.478 
      --experiment_name $NAME
      --setting $SETTING
      --repeat_time 1
      --reset_img_feat 1 
      --features $IMGFEAT
      --img_feat_mode $IMGFEATMODE
      --img_feat_pattern $IMGFEATPATTERN
      --max_episode_len 40 --K 20 --logit --beam"

CUDA_VISIBLE_DEVICES=$CUDAID python test_with_reranker.py $flag 
