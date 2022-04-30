CUDAID=${1:-0}
IMGFEAT=${2:-places365}
MODEL=VLNBERT-train-OriginalR2R
NAME=${IMGFEAT}_default

flag="--vlnbert oscar
      --submit 0
      --test_only 0
      --train validlistener
      --load snap/$MODEL/state_dict/best_val_unseen
      --name $NAME
      --features $IMGFEAT
      --maxAction 15
      --batchSize 8
      --feedback sample
      --lr 1e-5
      --iters 300000
      --optim adamW
      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5"

CUDA_VISIBLE_DEVICES=$CUDAID python r2r_src/train.py $flag