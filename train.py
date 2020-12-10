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

########## Data augmetation ##########
# Augmetation function
def build_sem_seg_train_aug(cfg):
    augs = [T.RandomBrightness(0.5, 2),
            T.RandomContrast(0.5, 2),
            T.RandomSaturation(0.5, 2),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True)]
    return augs


# Custom trainer
class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

########## Load data ##########
# register training and test datasets to Detectron2
register_coco_instances("T", {}, "/content/gdrive/My Drive/pascal_train.json", "/content/gdrive/My Drive/train_images")
register_coco_instances("V", {}, "/content/gdrive/My Drive/test.json", "/content/gdrive/My Drive/test_images")

# metadata
train_metadata = MetadataCatalog.get("T")
test_metadata = MetadataCatalog.get("V")

# dataset dictionary
train_dataset_dicts = DatasetCatalog.get("T")
test_dataset_dicts = DatasetCatalog.get("V")

########## Set cfg file ##########
cfg = get_cfg()

# load ImageNet pretrained weights
# cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

# load dataset
cfg.DATASETS.TRAIN = ("T")

# parameters
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 20000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20

########## Training ##########
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
