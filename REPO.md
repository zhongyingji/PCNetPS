
It is the implementation of Points-in-Context Network (PCNet). PCNet aims at utilizing the contextual information to the largest extent based on one-stage detector. PCNet extracts the re-identification (ReID) feature based on points aggregation rather than feature cropping like RoIAlign. The code is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). 

### About the repository
- Inherited from maskrcnn-benchmark, it supports distributed data parallel training and testing with multiple GPUs and multiple machines.
- Training and testing on [PRW](https://github.com/liangzheng06/PRW-baseline) and [CUHK-SYSU](http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html) is supported.
- PCNet is supported. It extracts ReID feature by points aggregation based on one-stage detector like RetinaNet.
- It supports joint person search models based on two-stage detector like [OIM](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xiao_Joint_Detection_and_CVPR_2017_paper.pdf).



### Installation
Please firstly follow the offical installation of [maskrcnn-benchmark INSTALL.md](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md). This repository does not support the mixed precision training, so feel free to skip the installation of `apex`.
**NOTE:** If you meet some problems during the installation, you may find a solution in [issues of official maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/issues). 

````bash
git clone https://github.com/zhongyingji/PCNetPS.git
cd PCNetPS
python setup.py build develop
````

### Dataset Preparation

Make sure you have downloaded the dataset of person search PRW-v16.04.20 and CUHK-SYSU.
- Symlink the path to the dataset to `maskrcnn_benchmark/datasets/` as follows:
```bash
ln -s /path_to_prw_dataset/PRW-v16.04.20 maskrcnn_benchmark/datasets/PRW-v16.04.20
ln -s /path_to_cuhk_sysu_dataset/CUHK-SYSU maskrcnn_benchmark/datasets/CUHK-SYSU
```
- The training of [APNet](https://github.com/zhongyingji/APNet) requires the keypoint information:

````bash
cp info/prw/* maskrcnn_benchmark/datasets/PRW-v16.04.20
cp info/cuhk/* maskrcnn_benchmark/datasets/CUHK-SYSU/dataset
````

### Get Started
Examples of PCNet on PRW dataset. For the usage of more models, please refer to [GETSTARTED.md](https://github.com/zhongyingji/PCNetPS/blob/main/GETSTARTED.md).
#### Train
- Training with single GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
--config-file "configs/personsearch/pcnet/prw_retinanet_fast_R-50-FPN_P5.yaml" \
OUTPUT_DIR "models/prw_fast"
```
- When model or batch is too large to fit into a single GPU, execute the following scripts for **multi-gpus training**:
```bash
export $NGPUS=2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
--config-file "configs/personsearch/pcnet/prw_retinanet_fast_R-50-FPN_P5.yaml" \
OUTPUT_DIR "models/prw_fast"
```

#### Test
- After the training has finished, run the exactly same scripts to test the model. 
- `TEST.IMS_PER_BATCH` has defaulty been set to 1 for speed measure. **Multi-image batch testing** speeds up the inference process, with error less than 0.1% :
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
--config-file "configs/personsearch/pcnet/prw_retinanet_fast_R-50-FPN_P5.yaml" \
OUTPUT_DIR "models/prw_fast" TEST.IMS_PER_BATCH 8
```
The best performed models listed in paper are trained and tested with scripts in `scripts/train_pcnet_prw.sh`.

#### Todo
- [x] Visualize the points of PCNet
- [ ] APNet, BINet
- [ ] NAE
- [ ] PCNet based on FCOS
- [ ] Evaluation of the detection


