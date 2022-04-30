CUDAID=${1:-0}
features=vit
ft_dim=512
CHECKPOINT=../datasets/RxR/trained_models/vitbase.clip-finetune

ngpus=1
seed=0

outdir=../datasets/RxR/trained_models

flag="--root_dir ../datasets
      --output_dir ${outdir}
      
      --dataset RxR-en

      --resume_file ${CHECKPOINT}/best_val_unseen 
      --test 

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

      --lr 1e-5
      --iters 200000
      --log_every 2000
      --optim adamW

      --ml_weight 0.2
      --featdropout 0.4
      --dropout 0.5"

# inference
CUDA_VISIBLE_DEVICES=$CUDAID python r2r/main.py $flag 