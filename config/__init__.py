# ----------------------- Model Config -----------------------
from .fcos_config import fcos_cfg
from .pdetr_config import pdetr_cfg

def build_config(args):
    # FCOS
    if args.model in fcos_cfg.keys():
        return fcos_cfg[args.model]
    # Plain-DETR
    if args.model in pdetr_cfg.keys():
        return pdetr_cfg[args.model]
    
    else:
        print('Unknown Model: {}'.format(args.model))
        exit(0)
