CUDAID=${1:-0}
MODEL=rxr_vit/agent_rxr_en_clip_vit_maxinput160_ml04
IMGFEAT=vit
name=agent_${IMGFEAT}

flag="--attn soft
      --train validlistener 
      --load snap/$MODEL/state_dict/best_val_unseen
      --name $name 
      --features $IMGFEAT
      --featdropout 0.3
      --angleFeatSize 128
      --language en
      --maxInput 160
      --feature_size 512
      --feedback sample
      --mlWeight 0.4
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 80000 --maxAction 35"

CUDA_VISIBLE_DEVICES=$CUDAID python rxr_src/train.py $flag
