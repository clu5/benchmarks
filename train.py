from __future__ import annotations
from pathlib import Path
import click
from time import perf_counter
from tqdm import tqdm
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
    '-bs', '--batch_size', default=8, type=int, help='Batch size to use',
)
@click.option(
    '-a', '--architecture', 
    default='DenseNet', 
    type=click.Choice([
        'DenseNet',
        'ResNet',
        #'EfficientNet',
        #'SENet',
        'SEResNet50',
        'SEResNet101',
        'SEResNet152',
        #'HighResNet',
        #'ViT',
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
    '-lr', '--learning_rate', default=1e-4, type=float, help='Learning rate',
)
@click.option(
    '-e', '--epochs', default=20, type=int, help='Number of training epochs',
)
@click.option(
    '-s', '--seed', default=1234567890, type=int, help='Random seed',
)
@click.option(
    '-r', '--runs', default=1, type=int, help='Runs to estimate variance',
)
@click.option('-o', '--output_dir', default=Path.cwd(), help='Output directory',
)
@click.option(
    '--debug', is_flag=True, help='Use subset of data and print more verbosely',
)
def main(
        dataset: str, 
        batch_size: int, 
        architecture: str,
        block_config: list[int],
        learning_rate: float,
        epochs: int,
        seed: int,
        runs: int,
        output_dir: Path,
        debug: bool,
    ) -> None:
    if debug:
        stop_iterations = 256
        epochs = 2
        runs = 2

    use_3d = True if '3d' in dataset else False
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # use multiple model training runs to estimate variance
    for r in range(runs):
        print(f'RUN: {r}'.center(40))
        seed += 1
        monai.utils.set_determinism(seed)

        info = INFO[dataset]
        task = info['task']
        num_channels = info['n_channels']
        num_classes = len(info['label'])

        # make medmnist dataset
        DataClass = getattr(medmnist, info['python_class'])
        train_dataset = DataClass(split='train', download=True)
        valid_dataset = DataClass(split='val', download=True)
        test_dataset = DataClass(split='test', download=True)

        img_size = np.array(train_dataset[0][0]).shape
        if debug:
            print(f'img_size: {img_size}')

        # wrap dataset output in dictionary to use dictionary transforms
        class wrapper():
            def __init__(self, ds):
                self.ds = ds

            def __len__(self):
                return len(self.ds)

            def __getitem__(self, i):
                img, label = self.ds[i]
                return {'img': np.array(img), 'label': label}

        train_dataset = wrapper(ds=train_dataset)
        valid_dataset = wrapper(ds=valid_dataset)
        test_dataset = wrapper(ds=test_dataset)

        # wrap in monai dataset to be able to use 3D transforms
        transform = [
            monai.transforms.ScaleIntensityd(keys=['img']),
        ]
        if dataset in ('octmnist', 'pneumoniamnist', 'breastmnist', 'tissuemnist'):
            transform.append(monai.transforms.AddChanneld(keys=['img']))
        elif dataset in ('pathmnist', 'dermamnist', 'retinamnist', 'bloodmnist'):
            transform.append(monai.transforms.AsChannelFirstd(keys=['img'], channel_dim=-1))


        transform = monai.transforms.Compose(transform)

        monai_train_dataset = monai.data.IterableDataset(
            train_dataset, transform=transform,
        )
        monai_valid_dataset = monai.data.IterableDataset(
            valid_dataset, transform=transform,
        )
        monai_test_dataset = monai.data.IterableDataset(
            test_dataset, transform=transform,
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=monai_train_dataset, batch_size=batch_size, 
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=monai_valid_dataset, batch_size=2*batch_size, 
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=monai_test_dataset, batch_size=2*batch_size, 
        )

        # make model
        model_parameters = dict(
            spatial_dims=3 if use_3d else 2,
            in_channels=num_channels,
        )
        if architecture == 'DenseNet':
            model_parameters['block_config'] = block_config
            model_parameters['out_channels'] = num_classes
        else:
            model_parameters['num_classes'] = num_classes

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
            for i, batch_data in enumerate(tqdm(train_loader)):
                if debug and i > stop_iterations:
                    break

                x = batch_data['img'].cuda()
                y = batch_data['label'].cuda()
                optimizer.zero_grad()
                y_hat = model(x)
                if task == 'multi-label, binary-class':
                    y = y.to(torch.float32)
                else:
                    y = y.squeeze().long()
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()

                if task != 'multi-label, binary-class':
                    train_metrics['correct'] += (y_hat.argmax(-1) == y).sum().item()
                    train_metrics['total'] += y.shape[0]

            valid_metrics = {
                'correct': 0,
                'total': 0
            }

            model.eval()
            for batch_data in tqdm(valid_loader):
                if debug and i > stop_iterations:
                    break

                x = batch_data['img'].cuda()
                y = batch_data['label'].cuda()
                with torch.no_grad():
                    y_hat = model(x)
                    if task != 'multi-label, binary-class':
                        y = y.squeeze()

                if task != 'multi-label, binary-class':
                    valid_metrics['correct'] += (y_hat.argmax(-1) == y).sum().item()
                    valid_metrics['total'] += y.shape[0]

            if task != 'multi-label, binary-class':
                train_accuracy = train_metrics['correct'] / max(1, train_metrics['total'])
                valid_accuracy = valid_metrics['correct'] / max(1, valid_metrics['total'])

                print(f'{e}\tTRAIN\t{train_accuracy:.2f}')
                print(f'{e}\tVALID\t{valid_accuracy:.2f}')

        # evaluate model and save predictions 
        model.eval()
        with torch.no_grad():
            valid_y_true = torch.tensor([])
            valid_y_score = torch.tensor([])

            for batch_data in valid_loader:
                if debug and i > stop_iterations:
                    break

                x = batch_data['img'].cuda()
                y = batch_data['label'].cuda()
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
            np.save(output_dir / f'{save_exp}_true.npy', valid_y_true)
            np.save(output_dir / f'{save_exp}_score.npy', valid_y_score)

            valid_evaluator = Evaluator(dataset, 'val')
            valid_metrics = valid_evaluator.evaluate(valid_y_score)
            print('VALID'.center(20), valid_metrics)


            test_y_true = torch.tensor([])
            test_y_score = torch.tensor([])

            for batch_data in test_loader:
                if debug and i > stop_iterations:
                    break

                x = batch_data['img'].cuda()
                y = batch_data['label'].cuda()
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
    start_time = perf_counter()
    main()
    end_time = perf_counter()
    print(f'Runtime: {end_time - start_time:.0f}s')

