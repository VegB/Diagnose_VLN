CUDAID=${1:-0}
IMGFEATMODE=${2:-foreground}
SETTING=mask_env
IMGFEATPATTERN=CLIP-ViT-B-32-views_%s_m%.2f_%d.tsv

name=${SETTING}-${IMGFEAT}

features=vit  # which is 'vitbase_clip'
ft_dim=512

ngpus=1
seed=0

outdir=../datasets/RxR/trained_models

flag="--root_dir ../datasets
      --output_dir ${outdir}
      
      --dataset RxR-en

      --resume_file ../datasets/RxR/trained_models/vitbase.clip-finetune/best_val_unseen 
      --test 

      --setting ${SETTING}
      --repeat_time 1

      --ob_type pano
      --no_lang_ca

      --world_size ${ngpus}
      --seed ${seed}
      
      --fix_lang_embedding
      --fix_hist_embedding

      --num_l_layers 9
      --num_x_layers 4
      --hist_enc_pano
      --hist_pano_num_layers 2

      --features ${features}
      --feedback sample

      --max_action_len 20
      --max_instr_len 250
      --batch_size 8

      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --reset_img_feat 1
      --img_feat_pattern $IMGFEATPATTERN
      --img_feat_mode $IMGFEATMODE

      --lr 1e-5
      --iters 200000
      --log_every 2000
      --optim adamW

      --ml_weight 0.2
      --featdropout 0.4
      --dropout 0.5"

# inference
CUDA_VISIBLE_DEVICES=$CUDAID python r2r/main.py $flag
