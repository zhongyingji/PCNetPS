CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file "configs/personsearch/pcnet/cuhk_retinanet_fast_R-50-FPN_P5.yaml" \
OUTPUT_DIR "models/cuhk_fast"

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
--config-file "configs/personsearch/pcnet/cuhk_retinanet_multianchor_R-50-FPN_P5.yaml" OUTPUT_DIR "models/cuhk_multianchor" \
TEST.IMS_PER_BATCH 16

