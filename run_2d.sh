#! /bin/bash
python train.py --dataset pathmnist -a SEResNet50 -o output
python train.py --dataset dermamnist -a SEResNet50 -o output
python train.py --dataset octmnist -a SEResNet50 -o output
python train.py --dataset pneumoniamnist -a SEResNet50 -o output
python train.py --dataset retinamnist -a SEResNet50 -o output
python train.py --dataset breastmnist -a SEResNet50 -o output
python train.py --dataset bloodmnist -a SEResNet50 -o output
python train.py --dataset tissuemnist -a SEResNet50 -o output
