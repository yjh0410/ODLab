import json
import os
import contextlib
import torch
from datasets import build_dataset, build_transform

try:
    from pycocotools.cocoeval import COCOeval
except:
    print("It seems that the COCOAPI is not installed.")


class COCOAPIEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """
    def __init__(self, args, cfg, device, testset=False):
        # ----------------- Basic parameters -----------------
        self.ddp_mode = True if args.distributed else False
        self.image_set = 'test2017' if testset else 'val2017'
        self.device = device
        self.testset = testset
        # ----------------- Metrics -----------------
        self.map = 0.
        self.ap50_95 = 0.
        self.ap50 = 0.
        # ----------------- Dataset -----------------
        self.transform = build_transform(cfg, is_train=False)
        self.dataset, self.dataset_info = build_dataset(args, self.transform, is_train=False)


    @torch.no_grad()
    def evaluate(self, model):
        ids = []
        coco_results = []
        model.eval()
        model.trainable = False

        # start testing
        for index, (image, target) in enumerate(self.dataset):
            if index % 500 == 0:
                print('[Eval: %d / %d]'%(index, len(self.dataset)))
            # image id
            id_ = int(target['image_id'])
            ids.append(id_)
            
            # inference
            image = image.unsqueeze(0).to(self.device)
            outputs = model(image)
            bboxes, scores, cls_inds = outputs

            # rescale bbox
            orig_h, orig_w = target["orig_size"].tolist()
            bboxes[..., 0::2] *= orig_w
            bboxes[..., 1::2] *= orig_h
            
            # reformat results
            for i, box in enumerate(bboxes):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                label = self.dataset.coco_indexs[int(cls_inds[i])]
                
                # COCO json format
                bbox = [x1, y1, x2 - x1, y2 - y1]
                score = float(scores[i])
                A = {"image_id": id_,
                     "category_id": label,
                     "bbox": bbox,
                     "score": score}
                coco_results.append(A)

        model.train()
        model.trainable = True
        annType = ['segm', 'bbox', 'keypoints']
        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(coco_results) > 0:
            print('evaluating ......')
            cocoGt = self.dataset.coco
            if self.testset:
                json.dump(coco_results, open('coco_test-dev.json', 'w'))
                cocoDt = cocoGt.loadRes('coco_test-dev.json')
            else:
                # suppress pycocotools prints
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        cocoDt = cocoGt.loadRes(coco_results)
                        cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
                        cocoEval.params.imgIds = ids
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                # update mAP
                ap50_95, ap50 = cocoEval.stats[0], cocoEval.stats[1]
                print('ap50_95 : ', ap50_95)
                print('ap50 : ', ap50)
                self.map = ap50_95
                self.ap50_95 = ap50_95
                self.ap50 = ap50
            del coco_results
        else:
            print('No coco detection results !')

