import torch
import torch.nn.functional as F
from functools import reduce 
import torch.nn as nn


def simple_conv2d(m, k, **kwargs):
    return F.conv2d(m.view(1, *m.shape), k.view(1, *k.shape), **kwargs)


class KernelFinder(torch.nn.Module):
    """ Given a batch of images returns the convolution 
        kernel that should applied to each one of them
    """
    
    def __init__(self, kernel_size=(3, 2, 2)):
        super(KernelFinder, self).__init__()
        self.kernel_size = kernel_size
        
        
        # Given a batch of images return a feature set
        # from which the kernel will be computed
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
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
        
        
class ImageWiseConv2d(torch.nn.Module):
    """ Apply N kernel to a batch of N images """
    
    def forward(self, x):
        images, kernels = x
        data = [simple_conv2d(image, kernel) for image, kernel in zip(images, kernels)]
        return torch.cat(data)
        
        
class SampleModel(torch.nn.Module):
    """ Inspired by https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
        Instead of finding a transformation matrix to apply to the input
        we find a Kernel/filter for each image to be applied during the convolution 
    """
    def __init__(self):
        super(SampleModel, self).__init__()
        
        self.find_kernels = KernelFinder(kernel_size=(1, 2, 2))
        self.iw_conv1 = ImageWiseConv2d()
        
        # usual
        self.conv2 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20 * 4 * 4, 50)
        self.fc2 = nn.Linear(50, 10)

        
    def forward(self, x):
        kernels = self.find_kernels(x)
        x = self.iw_conv1((x, kernels))
        
        # usual
        x = F.relu(F.max_pool2d(x, 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        x = x.view(-1, 20 * 4 * 4)
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
            root='/tmp/', 
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
                    e, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            
check()
train(5)

