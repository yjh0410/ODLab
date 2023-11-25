# DETRX: Enhanced DETR with faster convergence speed
We are still optimizing this work, and currently only presenting interim results. Our goal is to optimize the performance of DETR as much as possible without introducing too many annoying tricks.

- Tricks:

- [x] Focal loss for classification.
- [x] Iterative refinement
- [x] Look forward twice
- [x] Hybrid Matching
- [ ] Feature pyramid network for hierarchical backbone
- [ ] Box reparameterization


- ImageNet-1K_V1 pretrained

| Model         |  scale     |  FPS  | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | Weight | Logs  |
| ------------- | ---------- | ----- | ---------------------- |  ---------------  | ------ | ----- |
| DETRX_R18_1x  |  800,1333  |       |         28.0           |         48.6      | [ckpt](https://github.com/yjh0410/ODLab/releases/download/detection_weights/detrx_r18_1x_coco.pth) | [log](https://github.com/yjh0410/ODLab/releases/download/detection_weights/DETRX-R18-1x.txt) |
| DETRX_R50_1x  |  800,1333  |       |                        |                   |  |  |


## Train DETRX
### Single GPU
Taking training **DETRX_R18_1x** on COCO as the example,
```Shell
python main.py --cuda -d coco --root path/to/coco -m detrx_r18_1x --batch_size 16 --eval_epoch 2
```

### Multi GPU
Taking training **DETRX_R18_1x** on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda -dist -d coco --root path/to/coco -m detrx_r18_1x --batch_size 16 --eval_epoch 2 
```

## Test DETRX
Taking testing **DETRX_R18_1x** on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m detrx_r18_1x --weight path/to/detrx_r18_1x.pth -vt 0.4 --show 
```

## Evaluate DETRX
Taking evaluating **DETRX_R18_1x** on COCO-val as the example,
```Shell
python main.py --cuda -d coco --root path/to/coco -m detrx_r18_1x --resume path/to/detrx_r18_1x.pth --eval_first
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m detrx_r18_1x --weight path/to/weight -vt 0.4 --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m detrx_r18_1x --weight path/to/weight -vt 0.4 --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m detrx_r18_1x --weight path/to/weight -vt 0.4 --show --gif
```