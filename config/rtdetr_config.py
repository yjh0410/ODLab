# Real-time DETR

rtdetr_cfg = {
    'rtdetr_r18':{
        # ---------------- Model config ----------------
        ## Model scale
        # Backbone
        'backbone': 'resnet18',
        'backbone_norm': 'BN',
        'res5_dilation': False,
        'pretrained': True,
        'pretrained_weight': 'imagenet1k_v1',
        'freeze_at': -1,  # freeze none layer of the backbone
        'out_stride': [8, 16, 32],
        'max_stride': 32,
        'hidden_dim': 256,
        # Transformer Ecndoer
        'neck': 'hybrid_encoder',
        'fpn_num_blocks': 3,
        'fpn_expansion': 0.5,
        'fpn_act': 'relu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'en_num_heads': 8,
        'en_num_layers': 1,
        'en_ffn_dim': 1024,
        'en_dropout': 0.0,
        'pe_temperature': 10000.,
        'en_act': 'gelu',
        'en_pre_norm': False,
        # Transformer Decoder
        'transformer': 'rtdetr_transformer',
        'de_num_heads': 8,
        'de_num_layers': 3,
        'de_ffn_dim': 1024,
        'de_dropout': 0.0,
        'de_act': 'relu',
        'de_pre_norm': False,
        'de_num_points': 4,
        'num_queries': 300,
        'learnt_init_query': False,
        'pe_temperature': 10000.,
        'dn_num_denoising': 100,
        'dn_label_noise_ratio': 0.5,
        'dn_box_noise_scale': 1,
        # Post process
        'train_topk': 300,
        'train_conf_thresh': 0.001,
        'train_nms_thresh': 0.5,
        'test_topk': 300,
        'test_conf_thresh': 0.001,
        'test_nms_thresh': 0.5,
        'nms_class_agnostic': True,  # We prefer to use class-agnostic NMS in the demo.
        # ---------------- Assignment config ----------------
        'matcher_hpy': {'cost_class': 2.0,
                        'cost_bbox':  5.0,
                        'cost_giou':  2.0,},
        # ---------------- Loss config ----------------
        'loss_coeff': {'class': 1.0,
                       'bbox':  5.0,
                       'giou':  2.0,},
        # ----------------- Training -----------------
        ## Optimizer
        'optimizer': 'adamw',
        'base_lr': 0.0001 / 16,
        'backbone_lr_ratio': 0.1,
        'momentum': None,
        'weight_decay': 0.0001,
        'clip_max_norm': 0.1,
        ## Params dict
        'param_dict_type': 'detr',
        'lr_backbone_names': ['backbone',],
        'lr_linear_proj_names': ["reference_points", "sampling_offsets",],  # These two names are not required by PlainDETR
        'lr_linear_proj_mult': 0.1,
        'wd_norm_names': ["norm", "bias", "level_embed",],
        'wd_norm_mult': 0.0,
        ## LR Scheduler
        'lr_scheduler': 'step',
        'warmup': 'linear',
        'warmup_iters': 2000,
        'warmup_factor': 0.00066667,
        ## Model EMA
        'use_ema': True,
        'ema_decay': 0.9999,
        'ema_tau': 2000,
        ## Training scheduler
        'scheduler': '6x',
        'max_epoch': 72,      # 6x
        'lr_epoch': [66],     # 6x
        # ----------------- Input -----------------
        ## Transforms
        'train_min_size': [[640, 640]],   # short edge of image
        'train_max_size': 640,
        'test_min_size': [[640, 640]],
        'test_max_size': 640,
        'random_size': [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
        ## Pixel mean & std
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std':  [0.229, 0.224, 0.225],
        ## Transforms
        'detr_style': False,
        'trans_config': [
            {'name': 'RandomPhotometricDistort', 'prob': 0.5},
            {'name': 'RandomZoomOut', 'fill': [123.675, 116.28, 103.53]},
            {'name': 'RandomIoUCrop', 'prob': 0.8},
            {'name': 'RandomHFlip'},
            {'name': 'RandomResize'},
            {'name': 'RefineBBox', 'min_box_size': 1},
        ],
        'box_format': 'xywh',
        'normalize_coords': True,
    },

    'rtdetr_r50':{
        # ---------------- Model config ----------------
        ## Model scale
        # Backbone
        'backbone': 'resnet50',
        'backbone_norm': 'FrozeBN',
        'res5_dilation': False,
        'pretrained': True,
        'pretrained_weight': 'imagenet1k_v2',
        'freeze_at': 1,  # freeze stem layer + layer1 of the backbone
        'out_stride': [8, 16, 32],
        'max_stride': 32,
        'hidden_dim': 256,
        # Transformer Ecndoer
        'neck': 'hybrid_encoder',
        'fpn_num_blocks': 3,
        'fpn_expansion': 1.0,
        'fpn_act': 'relu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'en_num_heads': 8,
        'en_num_layers': 1,
        'en_ffn_dim': 1024,
        'en_dropout': 0.0,
        'pe_temperature': 10000.,
        'en_act': 'gelu',
        'en_pre_norm': False,
        # Transformer Decoder
        'transformer': 'rtdetr_transformer',
        'de_num_heads': 8,
        'de_num_layers': 6,
        'de_ffn_dim': 1024,
        'de_dropout': 0.0,
        'de_act': 'relu',
        'de_pre_norm': False,
        'de_num_points': 4,
        'num_queries': 300,
        'learnt_init_query': False,
        'pe_temperature': 10000.,
        'dn_num_denoising': 100,
        'dn_label_noise_ratio': 0.5,
        'dn_box_noise_scale': 1,
        # Post process
        'train_topk': 300,
        'train_conf_thresh': 0.001,
        'train_nms_thresh': 0.5,
        'test_topk': 300,
        'test_conf_thresh': 0.001,
        'test_nms_thresh': 0.5,
        'nms_class_agnostic': True,  # We prefer to use class-agnostic NMS in the demo.
        # ---------------- Assignment config ----------------
        'matcher_hpy': {'cost_class': 2.0,
                        'cost_bbox':  5.0,
                        'cost_giou':  2.0,},
        # ---------------- Loss config ----------------
        'loss_coeff': {'class': 1.0,
                       'bbox':  5.0,
                       'giou':  2.0,},
        # ----------------- Training -----------------
        ## Optimizer
        'optimizer': 'adamw',
        'base_lr': 0.0001 / 16,
        'backbone_lr_ratio': 0.1,
        'momentum': None,
        'weight_decay': 0.0001,
        'clip_max_norm': 0.1,
        ## Params dict
        'param_dict_type': 'detr',
        'lr_backbone_names': ['backbone',],
        'lr_linear_proj_names': ["reference_points", "sampling_offsets",],  # These two names are not required by PlainDETR
        'lr_linear_proj_mult': 0.1,
        'wd_norm_names': ["norm", "bias", "level_embed",],
        'wd_norm_mult': 0.0,
        ## LR Scheduler
        'lr_scheduler': 'step',
        'warmup': 'linear',
        'warmup_iters': 2000,
        'warmup_factor': 0.00066667,
        ## Model EMA
        'use_ema': True,
        'ema_decay': 0.9999,
        'ema_tau': 2000,
        ## Training scheduler
        'scheduler': '6x',
        'max_epoch': 72,      # 6x
        'lr_epoch': [66],     # 6x
        # ----------------- Input -----------------
        ## Transforms
        'train_min_size': [[640, 640]],   # short edge of image
        'train_max_size': 640,
        'test_min_size': [[640, 640]],
        'test_max_size': 640,
        'random_size': [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
        ## Pixel mean & std
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std':  [0.229, 0.224, 0.225],
        ## Transforms
        'detr_style': False,
        'trans_config': [
            {'name': 'RandomPhotometricDistort', 'prob': 0.5},
            {'name': 'RandomZoomOut', 'fill': [123.675, 116.28, 103.53]},
            {'name': 'RandomIoUCrop', 'prob': 0.8},
            {'name': 'RandomHFlip'},
            {'name': 'RandomResize'},
            {'name': 'RefineBBox', 'min_box_size': 1},
        ],
        'box_format': 'xywh',
        'normalize_coords': True,
    },

}