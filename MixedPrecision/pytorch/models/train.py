import torch
import torch.nn.functional as F

from MixedPrecision.pytorch.models.classifiers import HOConvClassifier, ConvClassifier, SpatialTransformerClassifier
from MixedPrecision.pytorch.models.optimizers import WindowedSGD


def train(models, epochs, dataset, olr, lr_reset_threshold=1e-05):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = torch.utils.data.DataLoader(
        batch_size=64,
        shuffle=True,
        num_workers=4,
        dataset=dataset
    )

    dataset_size = len(train_loader)
    models_optim = []

    for model in models:
        model = model.to(device)
        optimizer = WindowedSGD(
            model.parameters(),
            epoch_steps=dataset_size,
            window=dataset_size,
            lr_min=lr_reset_threshold,
            lr=olr)

        model.train()
        models_optim.append((model, optimizer))

    costs = []
    for e in range(0, epochs):
        all_cost = [0] * len(models_optim)

        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)

            for mid, (model, optimizer) in enumerate(models_optim):
                optimizer.zero_grad()

                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()

                all_cost[mid] += loss.item()
                optimizer.step(loss)

        print(f'Train {e:3d}/{epochs:3d} {all_cost} {models_optim[0][1].lr} ')
        costs.append(all_cost)

    return costs


if __name__ == '__main__':
    from torchvision import datasets, transforms

    ishape = (1, 28, 28)
    models = [
        ConvClassifier(input_shape=ishape),
        SpatialTransformerClassifier(ConvClassifier(ishape), input_shape=ishape),
        HOConvClassifier(input_shape=ishape),
        SpatialTransformerClassifier(HOConvClassifier(ishape), input_shape=ishape),
    ]

    mnist = datasets.MNIST(
        root='/tmp',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    train(models, 100, mnist, 0.01, 1e-05)



