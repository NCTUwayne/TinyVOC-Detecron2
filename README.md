# TinyVOC-Detecron2
NCTU VRDL HW3
## Hardware
* Google Colab
* GPU: Tesla T4
## Dataset
Tiny VOC dataset

contains 1,349 training images, 100 test images with 20 common object classes.
## Requirements
* Python = 3.6
* PyTorch 1.3
* torchvision
* OpenCV
* fvcore
* pycocotools
* cython
* GCC = 4.9
## Data preprocessing
See VRDL_HW3.ipyhb or train.py 'Data augmentation' part

Augmentation methods:

* T.RandomBrightness(0.5, 2)
* T.RandomContrast(0.5, 2)
* T.RandomSaturation(0.5, 2)
* T.RandomFlip(prob=0.5, horizontal=True, vertical=False)
* T.RandomFlip(prob=0.5, horizontal=False, vertical=True)
## Training
See VRDL_HW3.ipyhb or train.py

Mask R-CNN with ResNet-101, which is trained on top of ImageNet pre-trained weights ‘mask_rcnn_R_101_FPN_3x.yaml’.

Hyperparameters:
* Learning algorithm: Stochastic Gradient Descent (SGD)
* Batch size: 2
* Learning rate: 0.00025 
* The number of iterations: 20000
## Testing and output file
See VRDL_HW3.ipyhb or predict.py 'Make predictions' part 
## References
1. Detectron2: https://github.com/facebookresearch/detectron2.

2. Instance-Segmentation-using-Detectron2: https://github.com/AlessandroSaviolo/Instance-Segmentation-using-Detectron2.
