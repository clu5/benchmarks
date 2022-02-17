#! /bin/bash
python train.py --dataset organmnist3d -a "DenseNet" -o "output"
python train.py --dataset nodulemnist3d -a "DenseNet" -o "output"
python train.py --dataset adrenalmnist3d -a "DenseNet" -o "output"
python train.py --dataset fracturemnist3d -a "DenseNet" -o "output"
python train.py --dataset vesselmnist3d -a "DenseNet" -o "output"

python train.py --dataset organmnist3d -a "SEResNet50" -o "output"
python train.py --dataset nodulemnist3d -a "SEResNet50" -o "output"
python train.py --dataset adrenalmnist3d -a "SEResNet50" -o "output"
python train.py --dataset fracturemnist3d -a "SEResNet50" -o "output"
python train.py --dataset vesselmnist3d -a "SEResNet50" -o "output"

python train.py --dataset organmnist3d -a "SEResNet101" -o "output"
python train.py --dataset nodulemnist3d -a "SEResNet101" -o "output"
python train.py --dataset adrenalmnist3d -a "SEResNet101" -o "output"
python train.py --dataset fracturemnist3d -a "SEResNet101" -o "output"
python train.py --dataset vesselmnist3d -a "SEResNet101" -o "output"

python train.py --dataset organmnist3d -a "SEResNet152" -o "output"
python train.py --dataset nodulemnist3d -a "SEResNet152" -o "output"
python train.py --dataset adrenalmnist3d -a "SEResNet152" -o "output"
python train.py --dataset fracturemnist3d -a "SEResNet152" -o "output"
python train.py --dataset vesselmnist3d -a "SEResNet152" -o "output"
