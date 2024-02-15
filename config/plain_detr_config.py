# Plain DETR

plain_detr_cfg = {
    'rtpdetr_r50':{
        # ---------------- Model config ----------------
        ## Model scale
        # Backbone
        'backbone': 'resnet50',
        'backbone_norm': 'FrozeBN',
        'pretrained': True,
        'mae_pretrained': True,
        'max_stride': 32,
        'out_stride': 16,
        # Transformer Ecndoer
        'hidden_dim': 256,
        'en_num_heads': 8,
        'en_num_layers': 6,
        'en_ffn_dim': 2048,
        'en_dropout': 0.1,
        'en_act': 'gelu',
        # Transformer Decoder
        'transformer': 'plain_detr_transformer',
        'de_num_heads': 8,
        'de_num_layers': 6,
        'de_ffn_dim': 2048,
        'de_dropout': 0.0,
        'de_act': 'gelu',
        'de_pre_norm': True,
        'rpe_hidden_dim': 512,
        'use_checkpoint': False,
        'proposal_feature_levels': 3,
        'proposal_tgt_strides': [8, 16, 32],
        'num_queries_one2one': 300,
        'num_queries_one2many': 1500,
        # ---------------- Assignment config ----------------
        'matcher_hpy': {'cost_class': 2.0,
                        'cost_bbox': 1.0,
                        'cost_giou': 2.0,},
        # ---------------- Loss config ----------------
        'k_one2many': 6,
        'lambda_one2many': 1.0,
        'loss_coeff': {'class': 2,
                       'bbox': 1,
                       'giou': 2,},
        # ----------------- Training -----------------
        ## Optimizer
        'optimizer': 'adamw',
        'base_lr': 0.0002 / 16,
        'backbone_lr_ratio': 0.1,
        'momentum': None,
        'weight_decay': 0.05,
        'clip_max_norm': 0.1,
        ## LR Scheduler
        'lr_scheduler': 'step',
        'warmup': 'linear',
        'warmup_iters': 1000,
        'warmup_factor': 0.00066667,
        ## Training scheduler
        'scheduler': '1x',
        'max_epoch': 12,      # 1x
        'lr_epoch': [11],     # 1x
        # ----------------- Input -----------------
        ## Transforms
        'train_min_size': [800],   # short edge of image
        'train_min_size2': [400, 500, 600],
        'train_max_size': 1333,
        'test_min_size': 800,
        'test_max_size': 1333,
        'random_crop_size': [320, 600],
        ## Pixel mean & std
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std':  [0.229, 0.224, 0.225],
        ## Transforms
        'detr_style': True,
        'trans_config': None,
        'normalize_coords': False,
    },

}