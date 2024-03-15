# Re-TAG: Retrieval-Augmented Scene Graph Generation via Multi-Prototype Learning

We refer to the implementation of PE-Net. 

Check [INSTALL.md](./INSTALL.md) for installation instructions.

## Dataset
Check [DATASET.md](./DATASET.md) for instructions of dataset preprocessing.


## Train
ReTAG first train PE-Net, and then constructs memory bank.

If you want to skip the pre-training stage, and memory bank construction, please download the pre-trained model in this anonymous drive [link](https://).


After preparing the pre-trained PE-Net, and memory bank. You can train ReTAG using [scripts](./scripts/predcls_train_retag.sh)

```
bash scripts/predcls_train_retag.sh
```


## Test

We provide [scripts](./scripts/test.sh) for testing the models
```
bash script/test.sh
```


## Device

All our experiments are conducted on two NVIDIA GeForce RTX 3090 or one NVIDIA A6000, if you wanna run it on your own device, make sure to follow distributed training instructions in [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).




## Tips
We use the `rel_nms` [operation](./maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) provided by [RU-Net](https://github.com/siml3/RU-Net/blob/main/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) and [HL-Net](https://github.com/siml3/HL-Net/blob/main/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) in PredCls and SGCls to filter the predicted relation predicates, which encourages diverse prediction results. 


## Acknowledgement

The code is implemented based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).
