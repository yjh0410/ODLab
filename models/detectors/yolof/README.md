# YOLOF: You Only Look One-level Feature

- ImageNet-1K_V1 pretrained

| Model             |  scale     |  FPS  | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | Weight | Logs  |
| ----------------- | ---------- | ----- | ---------------------- |  ---------------  | ------ | ----- |
| YOLOF_R18_C5_1x   |  800,1333  |       |          32.8          |       51.2        | [ckpt](https://github.com/yjh0410/ODLab/releases/download/detection_weights/yolof_r18_c5_1x_coco.pth) | [log](https://github.com/yjh0410/ODLab/releases/download/detection_weights/YOLOF-R18-C5-1x.txt) |
| YOLOF_R50_C5_1x   |  800,1333  |       |          38.0          |       57.7        | [ckpt](https://github.com/yjh0410/ODLab/releases/download/detection_weights/yolof_r50_c5_1x_coco.pth) | [log](https://github.com/yjh0410/ODLab/releases/download/detection_weights/YOLOF-R50-C5-1x.txt) |
| YOLOF_R50_DC5_1x  |  800,1333  |       |                        |                   |  |  |
| YOLOF_R101_C5_1x  |  800,1333  |       |                        |                   |  |  |
| YOLOF_R101_DC5_1x |  800,1333  |       |                        |                   |  |  |

- ImageNet-1K_V2 pretrained

| Model             |  scale     |  FPS  | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | Weight | Logs  |
| ----------------- | ---------- | ----- | ---------------------- |  ---------------  | ------ | ----- |
| YOLOF_R50_C5_1x   |  800,1333  |       |                        |                   |  |  |
| YOLOF_R50_DC5_1x  |  800,1333  |       |                        |                   |  |  |
| YOLOF_R101_C5_1x  |  800,1333  |       |                        |                   |  |  |
| YOLOF_R101_DC5_1x |  800,1333  |       |                        |                   |  |  |

## Train YOLOF
### Single GPU
Taking training **YOLOF_R18_C5_1x** on COCO as the example,
```Shell
python main.py --cuda -d coco --root path/to/coco -m yolof_r18_c5_1x --batch_size 16 --eval_epoch 2
```

### Multi GPU
Taking training **YOLOF_R18_C5_1x** on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda -dist -d coco --root path/to/coco -m yolof_r18_c5_1x --batch_size 16 --eval_epoch 2 
```

## Test YOLOF
Taking testing **YOLOF_R18_C5_1x** on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m yolof_r18_c5_1x --weight path/to/yolof_r18_c5_1x.pth -vt 0.4 --show 
```

## Evaluate YOLOF
Taking evaluating **YOLOF_R18_C5_1x** on COCO-val as the example,
```Shell
python main.py --cuda -d coco --root path/to/coco -m yolof_r18_c5_1x --resume path/to/yolof_r18_c5_1x.pth --eval_first
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m yolof_r18_c5_1x --weight path/to/weight -vt 0.4 --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m yolof_r18_c5_1x --weight path/to/weight -vt 0.4 --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m yolof_r18_c5_1x --weight path/to/weight -vt 0.4 --show --gif
```