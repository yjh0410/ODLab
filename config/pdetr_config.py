# Plain-DETR config

pdetr_cfg = {
    'pdetr_r18_1x':{
        # ----------------- Model-----------------
        ## Backbone
        'backbone': 'resnet18',
        'backbone_norm': 'FrozeBN',
        'res5_dilation': False,
        'pretrained': True,
        'max_stride': 32,
        'out_stride': 16,
        ## Transformer
        'transformer': 'vanilla_transformer',
        'd_model': 256,
        ### - Encoder -
        'num_encoder': 6,
        'encoder_num_head': 8,
        'encoder_mlp_ratio': 4.0,
        'encoder_dropout': 0.1,
        'encoder_act': 'relu',
        ### - Decoder -
        'num_decoder': 6,
        'decoder_num_head': 8,
        'decoder_mlp_ratio': 4.0,
        'decoder_dropout': 0.0,
        'decoder_act': 'relu',
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
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        'loss_ctn_weight': 2.0,
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
        'warmup_iters': 500,
        'warmup_factor': 0.00066667,
        ## Epoch
        'max_epoch': 12,      # 1x
        'lr_epoch': [10],     # 1x
        # ----------------- Input -----------------
        ## Image size
        'train_min_size': 800,
        'train_max_size': 1333,
        'test_min_size': 800,
        'test_max_size': 1333,
        ## Pixel mean & std
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std':  [0.229, 0.224, 0.225],
        ## Transforms
        'trans_config': [
            {'name': 'RandomResize', 'random_sizes': [800], 'max_size': 1333},
            {'name': 'RandomHFlip'},
        ],
    },
    
    'pdetr_r50_1x':{
        # ----------------- Model-----------------
        ## Backbone
        'backbone': 'resnet50',
        'backbone_norm': 'FrozeBN',
        'res5_dilation': False,
        'pretrained': True,
        'max_stride': 32,
        'out_stride': 16,
        ## Transformer
        'transformer': 'vanilla_transformer',
        'd_model': 256,
        ### - Encoder -
        'num_encoder': 6,
        'encoder_num_head': 8,
        'encoder_mlp_ratio': 4.0,
        'encoder_dropout': 0.1,
        'encoder_act': 'relu',
        ### - Decoder -
        'num_decoder': 6,
        'decoder_num_head': 8,
        'decoder_mlp_ratio': 4.0,
        'decoder_dropout': 0.0,
        'decoder_act': 'relu',
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
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        'loss_ctn_weight': 2.0,
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
        'warmup_iters': 500,
        'warmup_factor': 0.00066667,
        ## Epoch
        'max_epoch': 12,      # 1x
        'lr_epoch': [10],     # 1x
        # ----------------- Input -----------------
        ## Image size
        'train_min_size': 800,
        'train_max_size': 1333,
        'test_min_size': 800,
        'test_max_size': 1333,
        ## Pixel mean & std
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std':  [0.229, 0.224, 0.225],
        ## Transforms
        'trans_config': [
            {'name': 'RandomResize', 'random_sizes': [800], 'max_size': 1333},
            {'name': 'RandomHFlip'},
        ],
    },
    
}