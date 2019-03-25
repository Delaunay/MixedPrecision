import torch
import torch.nn.functional as F
from functools import reduce 
import torch.nn as nn


def simple_conv2d(m, k, **kwargs):
    """  m:        inC x  H x  W
         k: outC x inC x kH x kW """
    
    x = F.conv2d(m.view(1, *m.shape), k, **kwargs)
    n, c, h, w = x.shape
    return x.view(c, h, w)


def test_simple_conv2d():
    a = torch.rand(3, 10, 10)
    k = torch.rand(1, 3, 2, 2)
    assert simple_conv2d(a, k).shape == (1, 9, 9)
    
    a = torch.rand(3, 10, 10)
    k = torch.rand(12, 3, 2, 2)
    assert simple_conv2d(a, k).shape == (12, 9, 9)
    
test_simple_conv2d()


def conv2d_iwk(images, kernels, **kwargs):
    """ Apply N kernels to a batch of N images

        Images : N x inC x H x W
        Kernels: N x outC x inC x kH x kW
        Output : N x outC x size(Conv2d)
    """

    data = []
    for image, out_kernels in zip(images, kernels):
        out_channels = []

        val = simple_conv2d(image, out_kernels, **kwargs)

        c, h, w = val.shape

        data.append(val.view(1, c, h, w))

    return torch.cat(data)


def test_conv2d_iwk():
    out_channel = 14
    in_channel = 3
    batch_size = 4
    
    imgs = torch.rand(batch_size, in_channel, 10, 10)
    ks = torch.rand(batch_size, out_channel, in_channel, 2, 2)

    assert conv2d_iwk(imgs, ks).shape == (batch_size, out_channel, 9, 9)
    
test_conv2d_iwk()


class KernelFinder(torch.nn.Module):
    """ Given a batch of images returns the convolution
        kernel that should applied to each one of them
    """

    def __init__(self, in_channel, out_channel, kernel_size=(3, 2, 2)):
        super(KernelFinder, self).__init__()
        self.kernel_size = (out_channel, *kernel_size)

        # Given a batch of images return a feature set
        # from which the kernel will be computed
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Given the previous features for a fiven batch returns
        # the conv kernel that should be applied
        self.kernel_finder = nn.Sequential(
            nn.Linear(90, 32),
            nn.ReLU(True),
            nn.Linear(32, reduce(lambda x, y: x * y, self.kernel_size))
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 10 * 3 * 3)
        x = self.kernel_finder(x)
        x = x.view(-1, *self.kernel_size)
        return x


def test_KernelFinder():
    out_channel = 14
    in_channel = 3
    batch_size = 4
    kernel_size = (1, 2, 2)
    
    model = KernelFinder(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size)
    
    imgs = torch.rand(batch_size, in_channel, 28, 28)
    
    assert model(imgs).shape == (batch_size, out_channel, *kernel_size)
    
test_KernelFinder()


class SampleModel(torch.nn.Module):
    """ Inspired by https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
        Instead of finding a transformation matrix to apply to the input
        we find a Kernel/filter for each image to be applied during the convolution 
    """
    def __init__(self):
        super(SampleModel, self).__init__()
        self.find_kernels = KernelFinder(in_channel=1, out_channel=10, kernel_size=(1, 2, 2))
        
        # usual
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        
    def forward(self, x):
        kernels = self.find_kernels(x)
        x = conv2d_iwk(x, kernels)
        
        # usual
        x = F.relu(F.max_pool2d(x, 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
    
    
def check():
    test = SampleModel()
    # Batch of 4 images 3x28x28
    x = torch.rand(4, 1, 28, 28)
    test(x)
    x = torch.rand(64, 1, 28, 28)
    test(x)

def train(epoch):
    import torch.optim as optim
    import torchvision
    from torchvision import datasets, transforms
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(
        batch_size=64, 
        shuffle=True, 
        num_workers=1,
        dataset=datasets.MNIST(
            root='/tmp', 
            train=True, 
            download=True,
            transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
           ])
        )
    )
    
    model = SampleModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    dset_size = len(train_loader.dataset)
    
    for e in range(epoch):
        model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
       
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    e, batch_idx * len(data), dset_size,
                    100. * batch_idx / len(train_loader), loss.item()))
            
check()
train(5)



