# DETRX config


detrx_cfg = {
    # ----------------- ResNet backbone -----------------
    'detrx_r18_1x':{
        # ----------------- Model-----------------
        ## Backbone
        'backbone': 'resnet18',
        'backbone_norm': 'FrozeBN',
        'res5_dilation': False,
        'pretrained': True,
        'pretrained_weight': 'imagenet1k_v1',
        'max_stride': 128,
        'out_stride': [8, 16, 32, 64, 128],
        ## Transformer Encoder
        'transformer_encoder': 'detr_encoder',
        'd_model': 256,
        'num_encoder': 6,
        'encoder_num_head': 8,
        'encoder_mlp_ratio': 8.0,
        'encoder_dropout': 0.1,
        'encoder_act': 'relu',
        ## FPN Neck
        'neck': 'basic_fpn',
        'fpn_p6_feat': True,
        'fpn_p7_feat': True,
        'fpn_p6_from_c5': False,
        ## Transformer Decoder
        'transformer_decoder': 'detrx_decoder',
        'num_decoder': 6,
        'decoder_num_head': 8,
        'decoder_mlp_ratio': 8.0,
        'decoder_dropout': 0.0,
        'decoder_act': 'relu',
        'num_queries_one2one': 300,
        'num_queries_one2many': 1500,
        'k_one2many': 6,
        'look_forward_twice': True,
        ## Post-process
        'train_topk': 300,
        'test_topk': 300,
        # ----------------- Label Assignment -----------------
        'matcher': 'HungarianMatcher',
        'matcher_hpy':{'cost_cls_weight':  2.0,
                       'cost_box_weight':  5.0,
                       'cost_giou_weight': 2.0,
                       },
        # ----------------- Loss weight -----------------
        ## Loss hyper-parameters
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
        'loss_cls_weight':  1.0,
        'loss_box_weight':  5.0,
        'loss_giou_weight': 2.0,
        'one2many_loss_weight': 1.0,
        # ----------------- Training -----------------
        ## Training scheduler
        'scheduler': '1x',
        ## Optimizer
        'optimizer': 'adamw',
        'base_lr': 0.0001 / 16,
        'backbone_lr_ratio': 0.1,
        'momentum': None,
        'weight_decay': 1e-4,
        'clip_max_norm': 0.1,
        ## LR Scheduler
        'lr_scheduler': 'step',
        'warmup': 'linear',
        'warmup_iters': 100,
        'warmup_factor': 0.00066667,
        ## Epoch
        'max_epoch': 12,      # 1x
        'lr_epoch': [10],     # 1x
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
        'normalize_coords': True,
    },
    
}