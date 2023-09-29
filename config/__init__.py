# ----------------------- Model Config -----------------------
from .fcos_config import fcos_cfg

def build_config(args):
    # FCOS
    if args.model in fcos_cfg.keys():
        return fcos_cfg[args.model]
    
    else:
        print('Unknown Model: {}'.format(args.model))
        exit(0)
