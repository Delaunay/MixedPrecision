from MixedPrecision.tools.args import get_parser
from MixedPrecision.tools.loaders import load_dataset
from hybrid_classifier import HybridClassifier, HybridLoss

import torch
<<<<<<< HEAD
import torch.nn as nn
import torchvision.models.resnet as resnet
import torchvision.transforms as transforms


torch.set_num_threads(12)


class HybridLoss:
    def __init__(self, mode=0.1):
        self.classifier_loss_function = nn.CrossEntropyLoss()
        self.regression_loss_function = nn.L1Loss(reduction='elementwise_mean')
        self.mode = mode
        self.classifier_loss = 0
        self.regression_loss = 0
        self.to_image = transforms.ToPILImage()
        self.resize = transforms.Resize((32, 32))
        self.gray = transforms.Grayscale()
        self.to_tensor = transforms.ToTensor()

    def __call__(self, y, y_hat, x, x_hat):
        # x_hat has only one channel

        # return self.mode * self.classifier_loss(y_hat, y) + (1 - self.mode) * ((x_hat - x_gray) ** 2)
        self.classifier_loss = self.mode * self.classifier_loss_function(y_hat, y)

        # This is awful but since we upsample CIFAR10 and that torch does not give us access to the original img...
        # this greatly help training since we only reconstruct a 32x32 img and not a 224x224!!
        x = x.cpu().view(-1, 3, 224, 224)
        xgpu = torch.tensor((), dtype=torch.float)
        xgpu = xgpu.new_zeros((len(x), 3, 32, 32)).cuda()

        for i, img in enumerate(x):
            xgpu[i] = self.to_tensor(self.resize(self.to_image(img + 0.5))).cuda()

        self.regression_loss = (1 - self.mode) * self.regression_loss_function(x_hat, xgpu)  #((x_hat - x_gray) ** 2).mean()

        v = self.classifier_loss + self.regression_loss

        self.classifier_loss = self.classifier_loss.item()
        self.regression_loss = self.regression_loss.item()
        return v
=======
>>>>>>> 0566951ffd6b02a30b6a5af152a1aeeb5483a7ca



<<<<<<< HEAD
class HybridClassifier(nn.Module):

    def __init__(self, classes=1000, output=(32, 32)):
        super(HybridClassifier, self).__init__()
        # self.resizer = nn.Linear(3 * 32 * 32, 3 * 224 * 224)
        # torch.nn.ConvTranspose2d()

        self.rectify_fc2_backward = []
        self.classify_fc_backward = []

        resnet18 = [2, 2, 2, 2]
        resnet50 = [3, 4, 6, 3]

        self.resnet = TruncatedResNet(resnet.BasicBlock, resnet50)
        self.classify_fc = nn.Linear(512 * resnet.BasicBlock.expansion, classes)
        # self.classify_fc.register_backward_hook(self.classify_callback)

        # torch.Size([64, 512, 7, 7]) => torch.Size([64, 8, 15, 15])
        self.rectify_fc1 = nn.ConvTranspose2d(512, 8, kernel_size=3, stride=2)
        # torch.Size([64, 8, 15, 15]) => 64, 1, 224, 224
        self.rectify_fc2 = nn.Linear(8 * 15 * 15, 3 * output[0] * output[1])
        # self.rectify_fc2.register_backward_hook(self.rectify_callback)

        self.output = output
        self.name = 'Experience1b'

    def rectify_callback(self, module, input, output):
        elem = [None] * len(output)
        for i, val in enumerate(output):
            elem[i] = val.mean().item()
        self.rectify_fc2_backward = elem

    def classify_callback(self, module, input, output):
        elem = [None] * len(output)
        for i, val in enumerate(output):
            elem[i] = val.mean().item()
        self.classify_fc_backward = elem

    def forward(self, x):
        # x = self.resizer(x.view(-1, 3, 32, 32))
        out, last = self.resnet(x)
        y_hat = self.classify_fc(out)

        # print(last.shape) = torch.Size([64, 512, 7, 7]) => 25088
        # target => 224x224 => 50176
        x_hat = self.rectify_fc1(last)
        # print(x_hat.shape) => torch.Size([64, 3, 15, 15])

        # x_hat = self.rectify_fc2(x_hat)
        x_hat = self.rectify_fc2(x_hat.view(-1, 8 * 15 * 15))

        return y_hat, x_hat.view(-1, 3, self.output[0], self.output[1])
=======
torch.set_num_threads(12)
>>>>>>> 0566951ffd6b02a30b6a5af152a1aeeb5483a7ca


class Experience1b(Baseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.name = self.name = 'Experience1b'

    def after_batch(self, id, batch_context):
        # print('Rectify {} \nClassify {}'.format(self.model.rectify_fc2_backward, self.model.classify_fc_backward))
        self.stats.write('{data[epoch][id]:2d}, {data[id]:4d}, {data[acc]:.2f}, {data[loss]:.4f}, {data[loss2]:.4f}\n'.format(data=batch_context))
        print('[{data[epoch][id]:2d} |{data[id]:4d}] acc: {data[acc]:.2f}%  loss: {data[loss]:.4f} {data[loss2]:.4f}'.format(data=batch_context))

    def train_batch(self):
        self.chrono.start()
        index, (x, y) = self.fetch_input()

        x = x.cuda()
        y = y.cuda()

        self.chrono.start()
        y_hat, x_hat = self.model(x)
        loss = self.criterion(y, y_hat, x, x_hat)

        acc = (y_hat.max(dim=1)[1] == y).sum().item() / len(y)
        self.update_stat(acc, self.criterion.classifier_loss, self.criterion.regression_loss)

        self.chrono.start()
        self.backward(loss)

        self.chrono.end()

    def merge_images(self, a, b):
        import PIL

        images = [a, b]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = PIL.Image.new('RGB', (total_width, max_height))

        x_offset = 0

        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        return new_im

    def recursive_acc(self, x_hat):
        new_x = torch.tensor((), dtype=torch.float)
        new_x = new_x.new_zeros((64, 3, 224, 224)).cuda()

        for i, img in enumerate(x_hat):
            new_x[i, 0] = img
            new_x[i, 1] = img
            new_x[i, 2] = img

        y_hat2, _ = self.model(new_x)
        return (y_hat2.max(dim=1)[1] == y).sum().item() / len(y)

    def show_testset(self, args, with_embedded_label=True):
        import torchvision.transforms as transforms
        #import matplotlib
        #matplotlib.use('Agg')
        #import matplotlib.pyplot as pl

        test = load_dataset(args, train=True)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.count = 0
        self.acc = 0
        self.cost = 0
        self.acc_recursive = 0
        count = 0

        for index, (x, y) in enumerate(test):
            with torch.no_grad():
                x = x.cuda()
                y = y.cuda()

                y_hat, x_hat = self.model(x)
                loss = self.criterion(y, y_hat, x, x_hat)
                acc = (y_hat.max(dim=1)[1] == y).sum().item() / len(y)
                self.update_stat(acc, loss)

                if index == 0:
                    to_image = transforms.ToPILImage()
                    size = len(x_hat)

                    x_hat = x_hat.cpu().view(size, 3, 32, 32)
                    x = x.cpu().view(size, 3, 224, 224)

                    for i, img in enumerate(x_hat):
                        hat = to_image(img + 0.5).resize((224, 224))
                        original = to_image(x[i] + 0.5)

                        img = self.merge_images(original, hat)
                        img.save('images/{}_{}.png'.format(i, classes[y[i]]))

        print('Test Set [EmbeddedLabel: {}] acc: {acc:.2f}%  loss: {loss:.4f} acc2: {acc2:.2f}'.format(
            with_embedded_label,
            acc=self.acc * 100 / self.count,
            loss=self.cost / self.count,
            acc2=self.acc_recursive)
        )


if __name__ == '__main__':
    import sys
    import torch
    import torch.nn as nn
    import torch.nn.parallel

    import torch.optim

    from MixedPrecision.tools.optimizer import OptimizerAdapter
    import MixedPrecision.tools.utils as utils

    from apex.fp16_utils import network_to_half

    sys.stderr = sys.stdout

    parser = get_parser()
    args = parser.parse_args()

    torch.set_num_threads(args.workers)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    utils.set_use_gpu(args.gpu, not args.no_bench_mode)
    utils.set_use_half(args.half)
    utils.show_args(args)

    data_loader = load_dataset(args, train=True)

    model = utils.enable_cuda(HybridClassifier())

    if args.half:
        model = network_to_half(model)

    criterion = utils.enable_cuda(HybridLoss())

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    optimizer = OptimizerAdapter(
        optimizer,
        half=args.half,
        static_loss_scale=args.static_loss_scale,
        dynamic_loss_scale=args.dynamic_loss_scale
    )
    model.train()

    trainer = Experience1b(
        model=model,
        loader=data_loader,
        criterion=criterion,
        optimizer=optimizer
    )

    trainer.epoch_count = args.epochs
    trainer.load_model()
    #trainer.train()
    trainer.show_testset(args, False)

"""
    Resnet50 + Regression
    
    Train Set [49 | 772] acc: 95.31%  loss: 0.0131 0.0311
    Train Set [49]       acc: 97.56%  loss: 0.0002
    Test Set             acc: 98.44%  loss: 0.0365

       Stage , Average , Deviation ,    Min ,    Max , count 
 fetch_input ,  0.2176 ,    0.0475 , 0.0029 , 0.2527 , 39090 
     forward ,  0.2006 ,    0.0061 , 0.0535 , 0.2453 , 39090 
    backward ,  0.0136 ,    0.0024 , 0.0090 , 0.2809 , 39090 
       Total ,  0.4319 ,    0.0477 , 0.0679 , 0.5259 , 39090 
    
             Metric ,   Average , Deviation ,       Min ,       Max , count 
    temperature.gpu ,   67.7603 ,    0.9117 ,   63.0000 ,   70.0000 , 69770 
    utilization.gpu ,   78.6346 ,   20.3697 ,   13.0000 ,  100.0000 , 69770 
 utilization.memory ,   54.9997 ,   12.1643 ,    7.0000 ,   73.0000 , 69770 
       memory.total , 6075.0000 ,    0.0000 , 6075.0000 , 6075.0000 , 69770 
        memory.free , 1152.3804 ,   17.2144 ,  565.0000 , 3252.0000 , 69770 
        memory.used , 4922.6196 ,   17.2144 , 2823.0000 , 5510.0000 , 69770
"""

