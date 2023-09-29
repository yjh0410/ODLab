# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from .detectors.yolof.build import build_yolof


# build object detector
def build_model(args, cfg, device, num_classes=90, trainable=False):
    # YOLOF    
    if 'yolof' in args.model:
        model, criterion = build_yolof(cfg, device, num_classes, trainable)
        
    if trainable:
        # Load pretrained weight
        if args.pretrained is not None:
            print('Loading pretrained weight ...')
            checkpoint = torch.load(args.pretrained, map_location='cpu')
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            # model state dict
            model_state_dict = model.state_dict()
            # check
            for k in list(checkpoint_state_dict.keys()):
                if k in model_state_dict:
                    shape_model = tuple(model_state_dict[k].shape)
                    shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                    if shape_model != shape_checkpoint:
                        checkpoint_state_dict.pop(k)
                        print(k)
                else:
                    checkpoint_state_dict.pop(k)
                    print(k)

            model.load_state_dict(checkpoint_state_dict, strict=False)

        # keep training
        if args.resume is not None:
            print('keep training: ', args.resume)
            checkpoint = torch.load(args.resume, map_location='cpu')
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            model.load_state_dict(checkpoint_state_dict)

        return model, criterion

    else:      
        return model
    