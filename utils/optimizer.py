import torch
from torch import optim


def build_optimizer(optimizer_cfg, model, resume=None):
    print('==============================')
    print('Optimizer: {}'.format(optimizer_cfg['optimizer']))
    print('--base_lr: {}'.format(optimizer_cfg['base_lr']))
    print('--backbone_lr_ratio: {}'.format(optimizer_cfg['backbone_lr_ratio']))
    print('--momentum: {}'.format(optimizer_cfg['momentum']))
    print('--weight_decay: {}'.format(optimizer_cfg['weight_decay']))

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": optimizer_cfg['base_lr'] * optimizer_cfg['backbone_lr_ratio'],
        },
    ]

    if optimizer_cfg['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            params=param_dicts, 
            lr=optimizer_cfg['base_lr'],
            momentum=optimizer_cfg['momentum'],
            weight_decay=optimizer_cfg['weight_decay']
            )
                                
    elif optimizer_cfg['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            params=param_dicts, 
            lr=optimizer_cfg['base_lr'],
            weight_decay=optimizer_cfg['weight_decay']
            )
                                
    start_epoch = 0
    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("optimizer")
        optimizer.load_state_dict(checkpoint_state_dict)
        start_epoch = checkpoint.pop("epoch")
                                                        
    return optimizer, start_epoch
