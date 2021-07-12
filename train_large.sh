if [ $# -eq 1 ]; then
  python3 train_sampling.py --gpu $1 --dataset ogbn-papers100M --sampler node --node-budget 6000 --num-repeat 50 --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0
else
  echo "please specify GPU id"
fi

