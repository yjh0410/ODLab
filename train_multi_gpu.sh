# 8 GPUs
python -m torch.distributed.run \
                    --nproc_per_node=2 \
                    --master_port 1669 \
                    main.py \
                    --cuda \
                    -dist \
                    -d coco \
                    --root /data/datasets/COCO/ \
                    -m pdetr_r18_1x \
                    --batch_size 8 \
                    --eval_epoch 2 \
                    --find_unused_parameters \
                    # --resume weights/coco/pdetr_r18_1x/pdetr_r18_1x_best.pth
