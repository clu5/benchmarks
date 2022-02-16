from __future__ import annotations
from pathlib import Path
import click
import numpy as np
import torch
import torchvision
import monai
import medmnist
from medmnist import INFO, Evaluator


@click.command()
@click.option(
    '-d', '--dataset', 
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
    '-dl', '--download', default=True, help='Download data if true',
)
@click.option(
    '-n', '--normalize', default=(0, 1), type=(float, float),
    help='Normalize data with these parameters: (mean, std)',
)
@click.option(
    '-bs', '--batch_size', default=8, type=int, help='Batch size to use',
)
@click.option(
    '-a', '--architecture', 
    default='DenseNet', 
    type=click.Choice([
        'DenseNet',
        'DenseNet121',
        'DenseNet169',
        'DenseNet201',
        'DenseNet264',
        'EfficientNet',
        'SENet',
        'SEResNet50',
        'HighResNet',
        'ViT',
    ]),
    help='Choose a model architecture',
)
@click.option(
    '-bc', '--block_config', 
    default=[6, 8, 12], 
    multiple=True,
    help='Number of blocks for DenseNet architecture',
)
@click.option(
    '-lr', '--learning_rate', default=1e-4, type=float, help='Set learning rate',
)
@click.option(
    '-e', '--epochs', default=10, type=int, help='Number of training epochs',
)
@click.option(
    '-s', '--seed', default=1234567890, type=int, help='Random seed',
)
@click.option(
    '-r', '--runs', default=1, type=int, help='How many runs to calculate variance',
)
@click.option(
    '-o', '--output_dir', 
    default=Path.cwd(), 
    help='Directory in which to write all results',
)
def main(
        dataset: str, 
        download: bool, 
        normalize: tuple[float, float], 
        batch_size: int, 
        architecture: str,
        block_config: list[int],
        learning_rate: float,
        epochs: int,
        seed: int,
        runs: int,
        output_dir: Path,
    ) -> None:

    # use multiple model training runs to estimate variance
    for r in range(runs):
        print(f'RUN: {r}'.center(40))
        seed += 1
        monai.utils.set_determinism(seed)

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
        if architecture == 'DenseNet':
            model_parameters['block_config'] = block_config

        model = getattr(monai.networks.nets, architecture)(**model_parameters)
        model = model.cuda()

        # optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if task == 'multi-label, binary-class':
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()

        # train model
        for e in range(epochs):
            train_metrics = {
                'correct': 0,
                'total': 0
            }
            model.train()
            for x, y in train_loader:
                x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()
                y_hat = model(x)
                if task != 'multi-label, binary-class':
                    y = y.squeeze()
                #breakpoint()
                loss = criterion(y_hat, y.long())
                loss.backward()
                optimizer.step()
                train_metrics['correct'] += (y_hat.argmax(-1) == y).sum().item()
                train_metrics['total'] += y.shape[0]

            valid_metrics = {
                'correct': 0,
                'total': 0
            }

            model.eval()
            for x, y in valid_loader:
                x, y = x.cuda(), y.cuda()
                with torch.no_grad():
                    y_hat = model(x)
                    if task != 'multi-label, binary-class':
                        y = y.squeeze()
                valid_metrics['correct'] += (y_hat.argmax(-1) == y).sum().item()
                valid_metrics['total'] += y.shape[0]

            train_accuracy = train_metrics['correct'] / train_metrics['total']
            valid_accuracy = valid_metrics['correct'] / valid_metrics['total']

            print(f'{e}\t{train_accuracy:.2f}')
            print(f'{e}\t{valid_accuracy:.2f}')

        # evaluate model and save predictions 
        model.eval()
        with torch.no_grad():
            valid_y_true = torch.tensor([])
            valid_y_score = torch.tensor([])

            for x, y in valid_loader:
                x, y = x.cuda(), y.cuda()
                y_hat = model(x)
                if task != 'multi-label, binary-class':
                    y = y.squeeze()
                    y = y.resize_(len(y), 1)

                score = y_hat.softmax(dim=-1)

                y = y.detach().cpu()
                score = score.detach().cpu()
                valid_y_true = torch.cat((valid_y_true, y), 0)
                valid_y_score = torch.cat((valid_y_score, score), 0)

            valid_y_true = valid_y_true.numpy()
            valid_y_score = valid_y_score.numpy()
            save_exp = f'{dataset}_{architecture}_{r}_valid'
            output_dir = Path(output_dir)
            np.save(output_dir / f'{save_exp}_true.npy', valid_y_true)
            np.save(output_dir / f'{save_exp}_score.npy', valid_y_score)

            #breakpoint()
            valid_evaluator = Evaluator(dataset, 'val')
            valid_metrics = valid_evaluator.evaluate(valid_y_score)
            print('VALID'.center(20), valid_metrics)


            test_y_true = torch.tensor([])
            test_y_score = torch.tensor([])

            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                y_hat = model(x)
                if task != 'multi-label, binary-class':
                    y = y.squeeze()
                    y = y.resize_(len(y), 1)

                score = y_hat.softmax(dim=-1)
                y = y.detach().cpu()
                score = score.detach().cpu()
                test_y_true = torch.cat((test_y_true, y), 0)
                test_y_score = torch.cat((test_y_score, score), 0)

            test_y_true = test_y_true.numpy()
            test_y_score = test_y_score.numpy()
            save_exp = f'{dataset}_{architecture}_{r}_test'
            np.save(output_dir / f'{save_exp}_true.npy', test_y_true)
            np.save(output_dir / f'{save_exp}_score.npy', test_y_score)

            test_evaluator = Evaluator(dataset, 'test')
            test_metrics = test_evaluator.evaluate(test_y_score)
            print('TEST'.center(20), test_metrics)

if __name__ == '__main__':
    main()

