CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
--config-file "configs/personsearch/oim/prw_R_50_C4.yaml" OUTPUT_DIR "models/prw_oim" SOLVER.IMS_PER_BATCH 4
