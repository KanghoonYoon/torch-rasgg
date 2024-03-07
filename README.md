# Re-TAG: Retrieval-Augmented Scene Graph Generation via Multi-Prototype Learning

This repository contains the official code implementation for the paper [Prototype-based Embedding Network for Scene Graph Generation](https://arxiv.org/abs/2303.07096).

## Installation
Check [INSTALL.md](./INSTALL.md) for installation instructions.

## Dataset
Check [DATASET.md](./DATASET.md) for instructions of dataset preprocessing.


## Train
We provide [scripts](./scripts/predcls/retag.sh) for training the models

```
bash script/predcls/retag.sh
```

## Test
We provide [scripts](./scripts/test.sh) for testing the models
```
bash script/test.sh
```


## Device

All our experiments are conducted on two NVIDIA GeForce RTX 3090 or one NVIDIA A6000, if you wanna run it on your own device, make sure to follow distributed training instructions in [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).



## The Trained Model Weights

We provide the weights for the  model. Due to random seeds and machines, they are not completely consistent with those reported in the paper, but they are within the allowable error range.

## Tips

We use the `rel_nms` [operation](./maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) provided by [RU-Net](https://github.com/siml3/RU-Net/blob/main/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) and [HL-Net](https://github.com/siml3/HL-Net/blob/main/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) in PredCls and SGCls to filter the predicted relation predicates, which encourages diverse prediction results. 


## Acknowledgement

The code is implemented based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

