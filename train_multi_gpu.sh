# 8 GPUs
python -m torch.distributed.run --nproc_per_node=2 --master_port 1663 main.py --cuda -dist -d coco --root /data/datasets/COCO/ -m fcos_r18_rt_3x --batch_size 16 --eval_epoch 2
