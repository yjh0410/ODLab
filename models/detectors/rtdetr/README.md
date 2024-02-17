# Real-time DETR

Our `RT-DETR` baseline on COCO-val:
```Shell
```

## Results on COCO

| Model         |  Scale     |  FPS  | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | Weight | Logs  |
| ------------- | ---------- | ----- | ---------------------- |  ---------------  | ------ | ----- |
| RT-DETR-R18   |  800,1333  |       |                        |                   |  |  |
| RT-DETR-R50   |  800,1333  |       |                        |                   |  |  |


## Train RT-DETR
### Single GPU
Taking training **RT-DETR** on COCO as the example,
```Shell
python main.py --cuda -d coco --root path/to/coco -m rtdetr_r50 --batch_size 16 --eval_epoch 2
```

### Multi GPU
Taking training **RT-DETR** on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda -dist -d coco --root path/to/coco -m rtdetr_r50 --batch_size 16 --eval_epoch 2 
```

## Test RT-DETR
Taking testing **RT-DETR** on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m rtdetr_r50 --weight path/to/rtdetr_r50.pth -vt 0.4 --show 
```

## Evaluate RT-DETR
Taking evaluating **RT-DETR** on COCO-val as the example,
```Shell
python main.py --cuda -d coco --root path/to/coco -m rtdetr_r50 --resume path/to/rtdetr_r50.pth --eval_first
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m rtdetr_r50 --weight path/to/weight -vt 0.4 --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m rtdetr_r50 --weight path/to/weight -vt 0.4 --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m rtdetr_r50 --weight path/to/weight -vt 0.4 --show --gif
```