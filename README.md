# Introduction


End-to-end pipeline data augmentation and training and evaluation script using PyTorch


## Data Augmentation


Generates different image transformation with many configurable parameters for scales, rotations, flips, crops and distortions with gaussian blurring, brightness and contrast variations.


## Automated Training


Automated flexible model selection and evaluation with different learning rates, learning rate decay, checkpoints and train test split.


# Usage

```
$ python striping.py


$ python train.py -h

usage: train.py [-h]
                [--model {alexnet,lenet5,stn-alexnet,stn-lenet5,capsnet,convneta,convnetb,convnetc,convnetd,convnete,convnetf,convnetg,convneth,convneti,convnetj,convnetk,convnetl,convnetm,convnetn,resnet18}]
                [--dataset {custom,cifar,hdf5}] [--num_classes NUM_CLASSES]
                [--batch_size BATCH_SIZE] [--transform TRANSFORM]
                [--num_workers NUM_WORKERS] [--lr-decay LR_DECAY]
                [--l2-reg L2_REG] [--hdf5-path HDF5_PATH]
                [--trainset-dir TRAINSET_DIR] [--testset-dir TESTSET_DIR]
                [--grey GREY]

PyTorch Automated Model Training & Evaluation

optional arguments:
  -h, --help            show this help message and exit
  --model {alexnet,lenet5,stn-alexnet,stn-lenet5,capsnet,convneta,convnetb,convnetc,convnetd,convnete,convnetf,convnetg,convneth,convneti,convnetj,convnetk,convnetl,convnetm,convnetn,resnet18}
  --dataset {custom,cifar,hdf5}
  --num_classes NUM_CLASSES
  --batch_size BATCH_SIZE
  --transform TRANSFORM
  --num_workers NUM_WORKERS
  --lr-decay LR_DECAY
  --l2-reg L2_REG
  --hdf5-path HDF5_PATH
  --trainset-dir TRAINSET_DIR
  --testset-dir TESTSET_DIR
  --grey GREY

```
