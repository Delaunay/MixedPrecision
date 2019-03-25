import torch
import torch.nn as nn
import torch.nn.parallel

import torch.optim
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):

    def __init__(self, in_channel, latent_size, ngf):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_size, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, in_channel, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):

    def __init__(self, in_channel, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channel, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class GAN:
    """
        GAN Training utility
    """

    def __init__(self, in_channel, latent_size, ngf, ndf, nz, batch_size, device, beta1, lr):
        """

            :param in_channel:
            :param latent_size:
            :param ngf:
            :param ndf:
            :param nz:
            :param batch_size:
            :param device:
            :param beta1:
            :param lr:
        """
        super(GAN, self).__init__()
        self.device = device
        self.net_g = Generator(in_channel, latent_size, ngf)
        self.net_d = Discriminator(in_channel, ndf)
        self.nz = nz

        self.criterion = nn.BCELoss()

        self.fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
        self.real_label = 1
        self.fake_label = 0

        # setup optimizer
        self.optimizerD = optim.Adam(self.net_d.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(self.net_g.parameters(), lr=lr, betas=(beta1, 0.999))

    @staticmethod
    def args(parser=None):
        import argparse

        if parser is None:
            parser = argparse.ArgumentParser(description=GAN.__doc__)
        else:
            subparser = parser.add_subparsers()
            parser = subparser.add_parser('dgan', description=GAN.__doc__)

        parser.add_argument('--batch-size', '-b', type=int, default=128,
                            help='')

        parser.add_argument('--image-size', type=int, default=64,
                            help='the height / width of the input image to network')

        parser.add_argument('--nz', type=int, default=100,
                            help='size of the latent z vector')

        parser.add_argument('--ngf', type=int, default=64)

        parser.add_argument('--ndf', type=int, default=64)

        parser.add_argument('--lr', type=float, default=0.0002,
                            help='learning rate, default=0.0002')

        parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam. default=0.5')

        parser.add_argument('--ngpu', type=int, default=1,
                            help='number of GPUs to use')

        parser.add_argument('--net-g', default='',
                            help="path to net-g parameters (to continue training)")

        parser.add_argument('--net-d', default='',
                            help="path to net-d parameters (to continue training)")

        parser.add_argument('--outf', default=None,
                            help='folder to output images and model checkpoints')

        return parser

    @classmethod
    def from_args(cls, args):
        pass

    def batch_step(self, data):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        data = [d.to(self.device) for d in data]
        torch.cuda.synchronize()

        self.net_d.zero_grad()
        real_cpu = data[0].to(self.device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), self.real_label, device=self.device)

        output = self.net_d(real_cpu)

        errD_real = self.criterion(output, label)
        errD_real.backward()

        # D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
        fake = self.net_g(noise)
        label.fill_(self.fake_label)

        output = self.net_d(fake.detach())

        errD_fake = self.criterion(output, label)
        errD_fake.backward()

        # D_G_z1 = output.mean().item()
        # errD = errD_real + errD_fake

        self.optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.net_g.zero_grad()
        label.fill_(self.real_label)  # fake labels are real for generator cost

        output = self.net_d(fake)

        errG = self.criterion(output, label)
        errG.backward()

        # D_G_z2 = output.mean().item()
        self.optimizerG.step()


if __name__ == '__main__':

    args = GAN.args()

    print(args)

