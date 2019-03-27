import torch
import torch.nn.functional as F

from MixedPrecision.pytorch.models.classifiers import HOConvClassifier, ConvClassifier, SpatialTransformerClassifier
from MixedPrecision.pytorch.models.optimizers import WindowedSGD

from benchutils.chrono import MultiStageChrono



if __name__ == '__main__':
    from torchvision import datasets, transforms
    import glob

    output = '/Tmp/pytorch'
    init_path = '/Tmp/pytorch'
    is_saved_init = False

    ishape = (1, 28, 28)
    models = {
        'conv': ConvClassifier(input_shape=ishape),
        'spatial_conv': SpatialTransformerClassifier(ConvClassifier(ishape), input_shape=ishape),
        'HO_conv': HOConvClassifier(input_shape=ishape),
        'spatial_HO': SpatialTransformerClassifier(HOConvClassifier(ishape), input_shape=ishape),
    }

    for name, model in models.items():
        vals = glob.glob(f'{init_path}/{name}_init')

        if len(vals) == 1:
            model.load_state_dict(torch.load(vals[0]))
