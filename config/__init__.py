# ----------------------- Model Config -----------------------
from .yolof_config import yolof_cfg
from .fcos_config import fcos_cfg

def build_config(args):
    # YOLOF
    if args.model in yolof_cfg.keys():
        return yolof_cfg[args.model]
    # FCOS
    elif args.model in fcos_cfg.keys():
        return fcos_cfg[args.model]
    
    else:
        print('Unknown Model: {}'.format(args.model))
        exit(0)
