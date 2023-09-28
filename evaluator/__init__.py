import os

from evaluator.coco_evaluator import COCOAPIEvaluator


def build_evluator(args, device, testset=False):
    # COCO Evaluator
    if args.dataset == 'coco':
        evaluator = COCOAPIEvaluator(args, device, testset)

    return evaluator
