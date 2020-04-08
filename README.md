# OptiFl-SCNN
## Not ready, test
## Reference
Fast-SCNN-pytorch : [https://github.com/Tramac/Fast-SCNN-pytorch](https://github.com/Tramac/Fast-SCNN-pytorch)

## Usage
1. Download Cityscapes dataset and unzip to './datasets/' folder
2. train
```
python train.py --model fast_scnn --dataset citys
```
3. flow
```
python flow.py (--datasets video_human --method farneback)
```