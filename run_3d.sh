#! /bin/bash
ARCH=${1:-"DenseNet"}

python train.py --dataset organmnist3d -a $ARCH --debug
python train.py --dataset nodulemnist3d -a $ARCH --debug
python train.py --dataset adrenalmnist3d -a $ARCH --debug
python train.py --dataset fracturemnist3d -a $ARCH --debug
python train.py --dataset vesselmnist3d -a $ARCH --debug
