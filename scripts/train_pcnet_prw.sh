CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file "configs/personsearch/pcnet/prw_retinanet_fast_R-50-FPN_P5.yaml" \
OUTPUT_DIR "models/prw_fast"

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
--config-file "configs/personsearch/pcnet/prw_retinanet_multianchor_R-50-FPN_P5.yaml" OUTPUT_DIR "models/prw_multianchor" \
TEST.IMS_PER_BATCH 2

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file "configs/personsearch/pcnet/prw_retinanet_R-50-FPN_P5.yaml" \
OUTPUT_DIR "models/prw_standard" 