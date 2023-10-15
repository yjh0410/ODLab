# ----------------------- Model Config -----------------------
from .retinanet_config import retinanet_cfg
from .fcos_config import fcos_cfg
from .yolof_config import yolof_cfg
from .pdetr_config import pdetr_cfg
from .pfcos_config import pfcos_cfg


def build_config(args):
    # RetinaNet
    if args.model in retinanet_cfg.keys():
        return retinanet_cfg[args.model]
    # FCOS
    elif args.model in fcos_cfg.keys():
        return fcos_cfg[args.model]
    # YOLOF
    elif args.model in yolof_cfg.keys():
        return yolof_cfg[args.model]
    # Plain-DETR
    elif args.model in pdetr_cfg.keys():
        return pdetr_cfg[args.model]
    # Plain-FCOS
    elif args.model in pfcos_cfg.keys():
        return pfcos_cfg[args.model]
    
    else:
        print('Unknown Model: {}'.format(args.model))
        exit(0)
