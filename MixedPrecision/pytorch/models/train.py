import torch
import torch.nn.functional as F

import MixedPrecision

from MixedPrecision.pytorch.models.classifiers import HOConvClassifier, ConvClassifier, SpatialTransformerClassifier
from MixedPrecision.pytorch.models.optimizers import WindowedSGD

from benchutils.chrono import MultiStageChrono


def train(models, epochs, dataset, olr, lr_reset_threshold=1e-05, output_name='/tmp/', device_name='gpu'):

    device = torch.device(device_name)

    train_loader = torch.utils.data.DataLoader(
        batch_size=64,
        shuffle=True,
        num_workers=4,
        dataset=dataset
    )

    dataset_size = len(train_loader)
    models_optim = {}

    for name, model in models.items():
        model = model.to(device)
        optimizer = WindowedSGD(
            model.parameters(),
            epoch_steps=dataset_size,
            window=dataset_size,
            lr_min=lr_reset_threshold,
            lr=olr)

        model.train()
        models_optim[name] = (model, optimizer)

    epoch_time = MultiStageChrono(name='train', skip_obs=10)
    costs = []
    for e in range(0, epochs):
        all_cost = [0] * len(models_optim)

        with epoch_time.time('epoch') as step_time:
            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(device), target.to(device)

                with epoch_time.time('models'):

                    for mid, (name, (model, optimizer)) in enumerate(models_optim.items()):

                        with epoch_time.time(model):
                            optimizer.zero_grad()

                            output = model(data)
                            loss = F.nll_loss(output, target)
                            loss.backward()

                            all_cost[mid] += loss.item()
                            optimizer.step(loss)

                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                        # ---
                    # ---
                # ---
            # ---

        with epoch_time.time('check_point'):
            for name, (model, _) in models_optim.items():
                torch.save(model.state_dict(), f'{output_name}/{name}_{e}')

        infos = [f'{all_cost[idx]:8.2f}, {models_optim[name][1].lr:10.8f}' for idx, name in enumerate(models_optim)]

        print(f'{e:3d}/{epochs:3d}, {step_time.val:6.2f}, ' + ', '.join(infos))

        costs.append(all_cost)

    print(epoch_time.to_json())
    return costs


if __name__ == '__main__':
    from torchvision import datasets, transforms
    import glob
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = 'cuda'
    if args.cpu or not torch.cuda.is_available():
        device = 'cpu'

    output = '/Tmp/pytorch'
    init_path = f'{"/".join(MixedPrecision.__file__.split("/")[:-1])}/pytorch/models/weights'

    is_saved_init = False

    ishape = (1, 28, 28)
    models = {
        'conv': ConvClassifier(input_shape=ishape),
        'spatial_conv': SpatialTransformerClassifier(ConvClassifier(ishape), input_shape=ishape),
        'HO_conv': HOConvClassifier(input_shape=ishape),
        'spatial_HO': SpatialTransformerClassifier(HOConvClassifier(ishape), input_shape=ishape),
    }

    # load init if available
    for name, model in models.items():
        vals = glob.glob(f'{init_path}/{name}_init')

        if len(vals) == 1:
            print(f'Loading {name}')
            is_saved_init = True
            model.load_state_dict(torch.load(vals[0]))

    # do not save init state if loaded from saved init
    if not is_saved_init:
        for name, model in models.items():
            torch.save(model.state_dict(), f'{output}/{name}_init')

    mnist = datasets.MNIST(
        root='/tmp',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    train(models, 100, mnist, 0.01, 1e-05, output, device)



