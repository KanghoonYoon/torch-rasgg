# RA-SGG: Retrieval-Augmented Scene Graph Generation Framework via Multi-Prototype Learning


We refer to the implementation of PE-Net [PENet](https://github.com/VL-Group/PENET?tab=readme-ov-file). 

Check [INSTALL.md](./INSTALL.md) for installation instructions.

## Dataset
Check [DATASET.md](./DATASET.md) for instructions of dataset preprocessing.


## Train
ReTAG requires the pre-trained PE-Net and memory bank, which is populated with the relation embedding of training dataset.

Please download [pre-trained models](https://drive.google.com/drive/folders/11jh-8F3LR8Hmm0Vsdp10Xc9PBbYQxnhK?usp=sharing) and the features for the [memory bank](https://drive.google.com/drive/folders/16sRrrmYfyK2jq12P0JUB7iXx8xG965tS?usp=sharing)

You can train ReTAG using [scripts](./scripts/predcls_train_retag.sh)

```
bash scripts/predcls_train_retag.sh
```


## Test

You can check the result of ReTAG in [Model_Zoo.md](./Model_Zoo.md)

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
