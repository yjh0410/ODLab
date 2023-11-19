# Dataset setting
DATASET="coco"
DATA_ROOT="/data/datasets/COCO/"
# DATA_ROOT="/Users/liuhaoran/Desktop/python_work/object-detection/dataset/COCO/"

# MODEL setting
MODEL="pdetr_r18_1x"
if [[ $MODEL == *"yolof"* ]]; then
    # Epoch setting
    BATCH_SIZE=64
    EVAL_EPOCH=2
elif [[ $MODEL == *"fcos"* ]]; then
    # Epoch setting
    BATCH_SIZE=16
    EVAL_EPOCH=2
elif [[ $MODEL == *"retinanet"* ]]; then
    # Epoch setting
    BATCH_SIZE=16
    EVAL_EPOCH=2
elif [[ $MODEL == *"pdetr"* ]]; then
    # Epoch setting
    BATCH_SIZE=16
    EVAL_EPOCH=2
else
    # Epoch setting
    BATCH_SIZE=16
    EVAL_EPOCH=2
fi

# -------------------------- Train Pipeline --------------------------
WORLD_SIZE=8
MASTER_PORT=1663
if [ $WORLD_SIZE == 1 ]; then
    python main.py \
        --cuda \
        --dataset ${DATASET}  \
        --root ${DATA_ROOT} \
        -m ${MODEL} \
        --batch_size ${BATCH_SIZE} \
        --eval_epoch ${EVAL_EPOCH} \
elif [[ $WORLD_SIZE -gt 1 && $WORLD_SIZE -le 8 ]]; then
    python -m torch.distributed.run --nproc_per_node=$WORLD_SIZE --master_port ${MASTER_PORT}  \
        main.py \
        --cuda \
        -dist \
        --dataset ${DATASET}  \
        --root ${DATA_ROOT} \
        -m ${MODEL} \
        --batch_size ${BATCH_SIZE} \
        --eval_epoch ${EVAL_EPOCH} \
        --find_unused_parameters \
else
    echo "The WORLD_SIZE is set to a value greater than 8, indicating the use of multi-machine \
          multi-card training mode, which is currently unsupported."
    exit 1
fi