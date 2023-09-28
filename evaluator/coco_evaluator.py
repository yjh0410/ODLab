import json
import tempfile
import torch
from datasets import build_dataset

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
    def __init__(self, args, device, testset=False):
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
        self.dataset = build_dataset(args, is_train=False)


    @torch.no_grad()
    def evaluate(self, model):
        ids = []
        data_dict = []
        num_images = len(self.dataset)
        model.eval()
        print('total number of images: %d' % (num_images))

        # start testing
        for index, (image, target) in enumerate(self.dataset):
            if index % 500 == 0:
                print('[Eval: %d / %d]'%(index, num_images))

            id_ = int(id_)
            ids.append(id_)
            
            # inference
            outputs = model(image)
            bboxes, scores, cls_inds = outputs

            # rescale bbox
            orig_h, orig_w = target["orig_size"].tolist()
            bboxes[..., 0::2] *= orig_w
            bboxes[..., 1::2] *= orig_h
            
            for i, box in enumerate(bboxes):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                label = self.dataset.class_ids[int(cls_inds[i])]
                
                bbox = [x1, y1, x2 - x1, y2 - y1]
                score = float(scores[i]) # object score * class score
                A = {"image_id": id_, "category_id": label, "bbox": bbox,
                     "score": score} # COCO json format
                data_dict.append(A)

        model.train()
        annType = ['segm', 'bbox', 'keypoints']
        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            print('evaluating ......')
            cocoGt = self.dataset.coco
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            if self.testset:
                json.dump(data_dict, open('coco_test-dev.json', 'w'))
                cocoDt = cocoGt.loadRes('coco_test-dev.json')
                return -1, -1
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, 'w'))
                cocoDt = cocoGt.loadRes(tmp)
                cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
                cocoEval.params.imgIds = ids
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()

                ap50_95, ap50 = cocoEval.stats[0], cocoEval.stats[1]
                print('ap50_95 : ', ap50_95)
                print('ap50 : ', ap50)
                self.map = ap50_95
                self.ap50_95 = ap50_95
                self.ap50 = ap50

                return ap50, ap50_95
        else:
            return 0, 0

