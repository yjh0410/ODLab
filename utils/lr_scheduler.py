import torch


# ------------------------- WarmUp LR Scheduler -------------------------
## Warmup LR Scheduler
class LinearWarmUpScheduler(object):
    def __init__(self, base_lr=0.01, wp_iter=500, warmup_factor=0.00066667):
        self.base_lr = base_lr
        self.wp_iter = wp_iter
        self.warmup_factor = warmup_factor


    def set_lr(self, optimizer, lr, base_lr):
        for param_group in optimizer.param_groups:
            init_lr = param_group['initial_lr']
            ratio = init_lr / base_lr
            param_group['lr'] = lr * ratio


    def __call__(self, iter, optimizer):
        # warmup
        alpha = iter / self.wp_iter
        warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        tmp_lr = self.base_lr * warmup_factor
        self.set_lr(optimizer, tmp_lr, self.base_lr)
        
## Build WP LR Scheduler
def build_wp_lr_scheduler(cfg, base_lr=0.01):
    print('==============================')
    print('WarmUpScheduler: {}'.format(cfg['warmup']))
    print('--base_lr: {}'.format(base_lr))
    print('--warmup_factor: {}'.format(cfg['warmup_factor']))
    print('--wp_iter: {}'.format(cfg['wp_iter']))

    if cfg['warmup'] == 'linear':
        wp_lr_scheduler = LinearWarmUpScheduler(base_lr, cfg['wp_iter'], cfg['warmup_factor'])
    
    return wp_lr_scheduler

                           
# ------------------------- LR Scheduler -------------------------
def build_lr_scheduler(cfg, optimizer, lr_epoch=None, resume=None):
    print('==============================')
    print('LR Scheduler: {}'.format(cfg['lr_scheduler']))

    if cfg['lr_scheduler'] == 'step':
        print('--lr_epoch: {}'.format(lr_epoch))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=cfg['lr_epoch'])
        
    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("lr_scheduler")
        lr_scheduler.load_state_dict(checkpoint_state_dict)

    return lr_scheduler
