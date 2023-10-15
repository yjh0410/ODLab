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
        'neck_expand_ratio': 0.25,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        ## Head
        'head': 'pfcos_head',
        'head_dim': 256,
        'num_cls_head': 4,
        'num_reg_head': 4,
        'head_act': 'relu',
        'head_norm': 'BN',
        'num_queries': 300,
        ## Post-process
        'train_topk': 100,
        'test_topk': 100,
        # ----------------- Label Assignment -----------------
        'matcher': 'HungarianMatcher',
        'matcher_hpy':{'cost_cls_weight': 2.0,
                       'cost_box_weight': 5.0,
                       'cost_giou_weight': 2.0,
                       },
        # ----------------- Loss weight -----------------
        ## Loss hyper-parameters
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
        'loss_cls_weight': 2.0,
        'loss_box_weight': 5.0,
        'loss_giou_weight': 2.0,
        # ----------------- Training -----------------
        ## Training scheduler
        'scheduler': '1x',
        ## Optimizer
        'optimizer': 'adamw',
        'base_lr': 0.0001 / 16,
        'backbone_lr_ratio': 1.0,
        'momentum': None,
        'weight_decay': 1e-4,
        'clip_max_norm': -1.0,
        ## LR Scheduler
        'lr_scheduler': 'step',
        'warmup': 'linear',
        'warmup_iters': 1500,
        'warmup_factor': 0.00066667,
        ## Epoch
        'max_epoch': 12,      # 1x
        'lr_epoch': [8, 11],  # 1x
        # ----------------- Input -----------------
        ## Transforms
        'train_min_size': [800],   # short edge of image
        'train_max_size': 1333,
        'test_min_size': 800,
        'test_max_size': 1333,
        ## Pixel mean & std
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std':  [0.229, 0.224, 0.225],
        ## Transforms
        'trans_config': [
            {'name': 'RandomHFlip'},
            {'name': 'RandomResize'},
        ],
        'normalize_coords': False,
    },
}