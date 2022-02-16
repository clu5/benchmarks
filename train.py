from __future__ import annotations
import pdb
import click
import numpy as np
import torch
import torchvision
import monai
import medmnist
from medmnist import INFO, Evaluator


@click.command()
@click.option(
    '--dataset', 
    default='breastmnist', 
    type=click.Choice([
        'pathmnist',
        'chestmnist',
        'dermamnist',
        'octmnist',
        'pneumoniamnist',
        'retinamnist',
        'breastmnist',
        'bloodmnist',
        'tissuemnist',
        'organamnist',
        'organcmnist',
        'organsmnist',
        'organmnist3d',
        'nodulemnist3d',
        'adrenalmnist3d',
        'fracturemnist3d',
        'vesselmnist3d',
        'synapsemnist3d',
    ]),
    help='Pick a MedMNIST dataset',
)
@click.option(
    '--download', default=True, help='Download data if true',
)
@click.option(
    '--normalize', default=(0, 1), type=(float, float),
    help='Normalize data with these parameters: (mean, std)',
)
@click.option(
    '--batch_size', default=8, type=int, help='Batch size to use',
)
@click.option(
    '--architecture', 
    default='DenseNet121', 
    type=click.Choice([
        'DenseNet121',
        'DenseNet169',
        'DenseNet201',
        'DenseNet264',
        'EfficientNet',
        'SEResNet50',
        'HighResNet',
        'ViT',
    ]),
    help='Choose a model architecture',
)
@click.option(
    '--learning_rate', default=1e-4, type=float, help='Set learning rate',
)
@click.option(
    '--epochs', default=10, type=int, help='Number of training epochs',
)
def main(
        dataset: str, 
        download: bool, 
        normalize: tuple[float, float], 
        batch_size: int, 
        architecture: str,
        learning_rate: float,
        epochs: int,
    ) -> None:
    info = INFO[dataset]
    task = info['task']
    num_channels = info['n_channels']
    num_classes = len(info['label'])

    mean, std = normalize
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])

    # make dataloaders
    DataClass = getattr(medmnist, info['python_class'])
    data_parameters = dict(download=download, transform=transform)
    train_data = DataClass(split='train', **data_parameters)
    valid_data = DataClass(split='val', **data_parameters)
    test_data = DataClass(split='test', **data_parameters)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_data, batch_size=2*batch_size, shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=2*batch_size, shuffle=False,
    )

    # make model
    model_parameters = dict(
        spatial_dims=3 if '3d' in dataset else 2,
        in_channels=num_channels,
        out_channels=num_classes,
    )

    model = getattr(monai.networks.nets, architecture)(**model_parameters)
    model = model.cuda()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    main()

