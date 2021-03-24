## Get started
This page includes the detailed training and testing tutorial of the repository.


### Training
- Training with single GPU:
```bash
 CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file ${CONFIG} OUTPUT_DIR ${DIR} [options]
 ```

- Training with multiple GPUs, 2 gpus for example: 
 ```bash
 export NGPUS=2
 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$NGPUS \
 tools/train_net.py \
--config-file ${CONFIG} OUTPUT_DIR ${DIR} [options]
 ```
#### Arguments
- `${CONFIG}`: config file of models, e.g., all files under `configs/personsearch/`, currently supports [OIM](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xiao_Joint_Detection_and_CVPR_2017_paper.pdf), [PCNet]()
- `${DIR}`: directory storing the trained model, e.g., `models/prw_pcnet_fast`
- `[options]`: adjust the configuration, e.g., `SOLVER.IMS_PER_BATCH 4`.

### Testing
After the training has done, run the exactly same commands for testing. Attaching ``[options]`` on the commands supports different purposes during testing:
- Multi-image batch testing for speedup, `TEST.IMS_PER_BATCH 8`
- Evaluate on CUHK-SYSU with different gallery sizes, `MODEL.REID_EVAL.NUM_CUHK_GALLERY 4000`
- Evaluate with groundtruth gallery boxes to check the performance upper bound, `MODEL.REID_EVAL.EVAL_WITH_GT_GALLERY True`
- Draw the detected boxes for visualization, `MODEL.REID_EVAL.DRAW_GALLERY_BOXES True`
- For **PCNet**, visualize the learned points by `MODEL.REID_EVAL.DRAW_PCNET_POINTS True`



### About the code
- For those who are interested in the codes, main codes of different models are organized as follows:
```bash
OIM: maskrcnn_benchmark/modeling/roi_heads/box_head
PCNet: maskrcnn_benchmark/modeling/rpn/retinanet
```
- One can also play around with the parameters `configs/personsearch/` to check the influence of parameter setting.

- The dataset processing and performance evaluation are in `maskrcnn_benchmark/data/datasets/prw(cuhk)_dataset.py`.

