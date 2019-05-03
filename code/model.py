import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Print(nn.Module):
    def forward(self,x):
        print('x.shape:', x.shape)
        return x

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu, nz=40, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        # self.main = nn.Sequential(
        #     # Two residual blocks
        #     ResidualBlock(nz, nz),
        #     ResidualBlock(nz, nz),
        #     ResidualBlock(nz, nz),
        #     ResidualBlock(nz, nz),
            
        #     # input is Z, going into a convolution
        #     nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(ngf * 8),
        #     # state size. 512 x 4 x 4
        #     nn.ConvTranspose2d( ngf * 8, ngf * 8, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. 512 x 4 x 4
        #     nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. 256 x 8 x 8
        #     nn.ConvTranspose2d(ngf * 4, ngf * 4, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. 256 x 8 x 8
        #     nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. 128 x 16 x 16
        #     nn.ConvTranspose2d( ngf * 2, ngf * 2, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. 128 x 16 x 16
        #     nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. 64 x 32 x 32
        #     nn.ConvTranspose2d( ngf, ngf, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(ngf),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. 64 x 32 x 32
        #     nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
        #     nn.Tanh()
        #     # state size. 3 x 64 x 64
        # )
        self.main = nn.Sequential(
            # Residual Blocks
            ResidualBlock(nz, nz),
            ResidualBlock(nz, nz),
            ResidualBlock(nz, nz),
            ResidualBlock(nz, nz),
            ResidualBlock(nz, nz),
            ResidualBlock(nz, nz),
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu, nz=40, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
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
            nn.LeakyReLU(0.2, inplace=True)
        )
        # self.main = nn.Sequential(
        #     # input is (nc) x 64 x 64
        #     nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # input is ndf x 32 x 32
        #     nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(ndf),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # input is ndf x 32 x 32
        #     nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*2) x 16 x 16
        #     nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*2) x 16 x 16
        #     nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*4) x 8 x 8
        #     nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*4) x 8 x 8
        #     nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*4) x 4 x 4
        #     nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 8),
        #     nn.LeakyReLU(0.2, inplace=True)
        #     # state size. (ndf*4) x 4 x 4
        # )

        # state size. (ndf*8) x 4 x 4
        # Used to classify whether real/fake
        self.conv_rf = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Used for domain classification
        self.conv_cls = nn.Conv2d(ndf * 8, nz, 4, 1, 0, bias=False)

    def forward(self, input):
        out = self.main(input)

        out_rf = self.conv_rf(out)
        out_rf = self.sigmoid(out_rf)

        out_cls = self.conv_cls(out)
        return out_rf.view(-1), out_cls.view(out_cls.size(0), out_cls.size(1))

