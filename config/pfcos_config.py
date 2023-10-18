# PlainFCOS: Fully Convolutional One-stage Object Detector dosen't need Multi-scale


pfcos_cfg = {
    # ImageNet1k-V1 pretrained
    'pfcos_r18_p5_1x':{
        # ----------------- Model-----------------
        ## Backbone
        'backbone': 'resnet18',
        'backbone_norm': 'FrozeBN',
        'res5_dilation': False,
        'pretrained': True,
        'pretrained_weight': 'imagenet1k_v1',
        'max_stride': 32,
        'out_stride': 32,
        ## Neck
        'neck': 'dilated_encoder',
        'neck_dilations': [2, 4, 6, 8],
        'neck_expand_ratio': 0.5,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        ## Head
        'head': 'pfcos_head',
        'head_dim': 256,
        'num_cls_head': 4,
        'num_reg_head': 4,
        'head_act': 'relu',
        'head_norm': 'BN',
        ## Post-process
        'train_topk': 100,
        'test_topk': 100,
        # ----------------- Label Assignment -----------------
        'matcher': 'ota',
        'matcher_hpy': {'topk_candidate': 1,
                        'sinkhorn_eps': 0.1,
                        'sinkhorn_iter': 50},
        # ----------------- Loss weight -----------------
        ## Loss hyper-parameters
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # ----------------- Training -----------------
        ## Training scheduler
        'scheduler': '1x',
        ## Optimizer
        'optimizer': 'adamw',
        'base_lr': 0.0001 / 16,
        'backbone_lr_ratio': 1.0 / 1.0,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'clip_max_norm': 10.0,
        ## LR Scheduler
        'lr_scheduler': 'step',
        'warmup': 'linear',
        'warmup_iters': 500,
        'warmup_factor': 0.00066667,
        ## Epoch
        'max_epoch': 12,      # 1x
        'lr_epoch': [8, 11],  # 1x
        # ----------------- Input -----------------
        ## Transforms
        'train_min_size': [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],   # short edge of image
        'train_min_size2': [400, 500, 600],
        'train_max_size': 1333,
        'test_min_size': 800,
        'test_max_size': 1333,
        'random_crop_size': [384, 600],
        ## Pixel mean & std
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std':  [0.229, 0.224, 0.225],
        ## Transforms
        'detr_style': True,
        'trans_config': None,
        'normalize_coords': False,
    },

}