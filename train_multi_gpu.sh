# 8 GPUs
python -m torch.distributed.run --nproc_per_node=2 --master_port 1663 main.py --cuda -dist -d coco --root /data/datasets/COCO/ -m pfcos_r18_p5_1x --batch_size 32 --eval_epoch 2
