# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader, DistributedSampler

from .coco import build_coco
from .transforms import build_transform


def build_dataset(args, transform=None, is_train=False):
    if args.dataset == 'coco':
        dataset = build_coco(args, transform, is_train)
        dataset_info = {
            'class_labels': dataset.coco_labels,
            'num_classes': 90
        }

    return dataset, dataset_info

def build_dataloader(args, dataset, collate_fn, is_train=False):
    if args.distributed:
        sampler = DistributedSampler(dataset) if is_train else DistributedSampler(dataset, shuffle=False)
    else:
        sampler = torch.utils.data.RandomSampler(dataset) if is_train else torch.utils.data.SequentialSampler(dataset)

    if is_train:
        batch_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=args.num_workers)
    else:
        dataloader = DataLoader(dataset, args.batch_size, sampler=sampler, drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers)
    
    return dataloader
