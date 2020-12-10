import torch, torchvision
import numpy as np
import pandas as pd
import random
import json
import cv2
import os

import detectron2
import detectron2.data.transforms as T
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger

import itertools
from itertools import groupby
from google.colab.patches import cv2_imshow

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskutil
from pycocotools import mask as maskUtils

setup_logger()


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle

########## Make predictions ##########
# load weights
cfg.MODEL.WEIGHTS = "/content/output/model_final.pth"
# define min score threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
# choose dataset
cfg.DATASETS.TEST = ("V")
# load training annotations
coco = COCO('/content/gdrive/My Drive/pascal_train.json')
# save categories
CLASS_NAMES = [coco.cats[k]['name'] for k in coco.cats.keys()]
# metadata
test_metadata.set(thing_classes=CLASS_NAMES)
# define predictor
predictor = DefaultPredictor(cfg)
# load test annotations
cocoGt = COCO("/content/gdrive/My Drive/test.json")
# store results
coco_dt = []

for imgid in cocoGt.imgs:
    # read test image
    image = cv2.imread("/content/gdrive/My Drive/test_images/" + cocoGt.loadImgs(ids=imgid)[0]['file_name'])[:,:,::-1]

    # make prediction
    outputs = predictor(image)

    # parse prediction
    boxes = (outputs['instances']._fields['pred_boxes'].tensor).cpu().numpy()
    scores = (outputs['instances']._fields['scores']).cpu().numpy()
    categories = (outputs['instances']._fields['pred_classes']).cpu().numpy()
    masks = (outputs['instances']._fields['pred_masks']).cpu().numpy()
    n_instances = len(scores)
    if len(categories) > 0:
        for i in range(n_instances):
            pred = {}
            pred['image_id'] = imgid
            pred['category_id'] = int(categories[i]) + 1
            pred['segmentation'] = binary_mask_to_rle(masks[i,:,:])
            pred['score'] = float(scores[i])
            coco_dt.append(pred)

with open('0856630.json', 'w') as f:
    f.write(pd.Series(coco_dt).to_json(orient='values'))
f.close()
