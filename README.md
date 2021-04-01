# GraphSAINT

This DGL example implements the sampling method proposed in the paper [GraphSAINT: Graph Sampling Based Inductive Learning Method](https://arxiv.org/abs/1907.04931).

Paper link: https://arxiv.org/abs/1907.04931

Author's code: https://github.com/GraphSAINT/GraphSAINT

Contributor: Liu Tang ([@lt610](https://github.com/lt610))

## Dependecies

- Python 3.7.0
- PyTorch 1.6.0
- numpy 1.19.2
- dgl 0.5.3

## Dataset

All datasets used are provided by Author's [code](https://github.com/GraphSAINT/GraphSAINT). They are available on [Google Drive link](https://drive.google.com/drive/folders/1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) (alternatively, [BaiduYun link (code: f1ao)](https://pan.baidu.com/s/1SOb0SiSAXavwAcNqkttwcg#list/path=%2F)). Dataset summary("m" stands for multi-class classifification, and "s" for single-class.):
| Dataset | Nodes | Edges | Degree | Feature | Classes | Train/Val/Test |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PPI | 14,755 | 225,270 | 15 | 50 | 121(m) | 0.66/0.12/0.22 |
| Flickr | 89250 | 899756 | 10 | 500 | 7(s) | 0.50/0.25/0.25 |
| Reddit | 232965 | 11606919 | 50 | 602 | 41(s) | 0.66/0.10/0.24 |
| Yelp | 716847 | 6877410 | 10 | 300 | 100(m) | 0.75/0.10/0.15 |
| Amazon | 1598960 | 132169734 | 83 | 200 | 107(m) | 0.85/0.05/0.10 |

## Full graph training

Run with following:
```bash
python train_full.py --gpu 0 --dataset ppi --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0 --batch-norm
python train_full.py --gpu 0 --dataset flickr --n-epochs 100 --n-hidden 256 --arch 1-1-0 --batch-norm --dropout 0.2
python train_full.py --gpu -1 --dataset reddit --n-epochs 100 --n-hidden 128 --arch 1-0-1-0 --batch-norm --dropout 0.1
python train_full.py --gpu -1 --dataset yelp --n-epochs 100 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
python train_full.py --gpu -1 --dataset amazon --n-epochs 100 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
```

## Minibatch training

Run with following:
```bash
python train_sampling.py --gpu 0 --dataset ppi --sampler node --node-budget 6000 --num-repeat 50 --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0 --batch-norm
python train_sampling.py --gpu 0 --dataset ppi --sampler edge --edge-budget 4000 --num-repeat 50 --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset ppi --sampler rw --num-roots 3000 --length 2 --num-repeat 50 --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset flickr --sampler node --node-budget 8000 --num-repeat 25 --n-epochs 100 --n-hidden 256 --arch 1-1-0 --batch-norm --dropout 0.2
python train_sampling.py --gpu 0 --dataset flickr --sampler edge --edge-budget 6000 --num-repeat 25 --n-epochs 100 --n-hidden 256 --arch 1-1-0 --batch-norm --dropout 0.2
python train_sampling.py --gpu 0 --dataset flickr --sampler rw --num-roots 6000 --length 2 --num-repeat 25 --n-epochs 100 --n-hidden 256 --arch 1-1-0 --batch-norm --dropout 0.2
python train_sampling.py --gpu 0 --dataset reddit --sampler node --node-budget 8000 --num-repeat 50 --n-epochs 100 --n-hidden 128 --arch 1-0-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset reddit --sampler edge --edge-budget 6000 --num-repeat 50 --n-epochs 100 --n-hidden 128 --arch 1-0-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset reddit --sampler rw --num-roots 2000 --length 4 --num-repeat 50 --n-epochs 100 --n-hidden 128 --arch 1-0-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset yelp --sampler node --node-budget 5000 --num-repeat 50 --n-epochs 100 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset yelp --sampler edge --edge-budget 2500 --num-repeat 50 --n-epochs 100 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset yelp --sampler rw --num-roots 1250 --length 2 --num-repeat 50 --n-epochs 100 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset amazon --sampler node --node-budget 4500 --num-repeat 50 --n-epochs 100 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset amazon --sampler edge --edge-budget 2000 --num-repeat 50 --n-epochs 100 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset amazon --sampler rw --num-roots 1500 --length 2 --num-repeat 50 --n-epochs 100 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
```

## Comparison

"Paper" means the results reported in paper. "Running" means the results of runing author's code. DGL means the results of the DGL implementation.

### F1-micro

#### Random node sampler

| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Paper | 0.960±0.001 | 0.507±0.001 | 0.962±0.001 |  |  |
| Running | 0.9628 | 0.5077 | 0.9622 |  |  |
| DGL | 0.5257 | 0.4943 | 0.8721 |  |  |

#### Random Edge sampler

| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Paper | 0.981±0.007 | 0.510±0.002 | 0.966±0.001 |  |  |
| Running | 0.9810 | 0.5066 | 0.9656 |  |  |
| DGL | 0.9147 | 0.5013 | 0.9243 |  |  |

#### Random Walk sampler
| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Paper | 0.981±0.004 | 0.511±0.001 | 0.966±0.001 |  |  |
| Running | 0.9812 | 0.5104 | 0.9648 |  |  |
| DGL | 0.9199 | 0.5045 | 0.8775 |  |  |

### Sample time

#### Random node sampler

| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Running | 1.0139 | 0.9574 | 9.0769 |  |  |
| DGL | 0.8725 | 1.1420 | 46.5929 | 68.4477 |  |

#### Random Edge sampler

| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Running | 0.8712 | 0.8764 | 4.7546 |  |  |
| DGL | 0.8635 | 1.0033 | 87.5684 |  |  |

#### Random Walk sampler

| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Running | 1.0880 | 1.7588 | 7.2055 |  |  |
| DGL | 0.7270 | 0.8973 | 58.1987 |  |  |
