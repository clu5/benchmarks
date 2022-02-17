#! /bin/bash
ARCH=${1:-"DenseNet"}
python train.py --dataset pathmnist -a SEResNet50 --debug
python train.py --dataset dermamnist -a SEResNet50 --debug
python train.py --dataset octmnist -a SEResNet50 --debug
python train.py --dataset pneumoniamnist -a SEResNet50  --debug
python train.py --dataset retinamnist -a SEResNet50 --debug
python train.py --dataset breastmnist -a SEResNet50 --debug
python train.py --dataset bloodmnist -a SEResNet50 --debug
python train.py --dataset tissuemnist -a SEResNet50 --debug
