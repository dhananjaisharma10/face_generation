import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# for debugging
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
    def __init__(self, in_dim=40, conv_dim=64, out_dim=3):
        super(Generator, self).__init__()
        self.feat_extractor = nn.Sequential(
            # Residual Blocks
            ResidualBlock(in_dim, in_dim),
            ResidualBlock(in_dim, in_dim),
            ResidualBlock(in_dim, in_dim),
            ResidualBlock(in_dim, in_dim),
            ResidualBlock(in_dim, in_dim),
            ResidualBlock(in_dim, in_dim),
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=in_dim, out_channels=conv_dim*8,
                                kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(conv_dim * 8),
            nn.ReLU(True),
            # state size. (conv_dim*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=conv_dim*8, out_channels=conv_dim*4,
                                kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim * 4),
            nn.ReLU(True),
            # state size. (conv_dim*4) x 8 x 8
            nn.ConvTranspose2d(in_channels=conv_dim*4, out_channels=conv_dim*2,
                                kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.ReLU(True),
            # state size. (conv_dim*2) x 16 x 16
            nn.ConvTranspose2d(in_channels=conv_dim*2, out_channels=conv_dim,
                                kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(True),
        )
        # state size. (conv_dim) x 32 x 32
        self.img_gen = nn.Sequential(
            nn.ConvTranspose2d(in_channels=conv_dim, out_channels=out_dim,
                                kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        # state size. (out_dim) x 64 x 64

    def forward(self, input):
        return self.img_gen(self.feat_extractor(input))

class Discriminator(nn.Module):
    def __init__(self, in_dim=3, conv_dim=64, label_dim=40):
        super(Discriminator, self).__init__()
        self.feat_extractor = nn.Sequential(
            # input is (in_dim) x 64 x 64
            nn.Conv2d(in_channels=in_dim, out_channels=conv_dim,
                        kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (conv_dim) x 32 x 32
            nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim*2,
                        kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (conv_dim*2) x 16 x 16
            nn.Conv2d(in_channels=conv_dim*2, out_channels=conv_dim*4,
                        kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (conv_dim*4) x 8 x 8
            nn.Conv2d(in_channels=conv_dim*4, out_channels=conv_dim*8,
                        kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (conv_dim*8) x 4 x 4
        # Used to classify whether real/fake (probability)
        self.prob = nn.Sequential(
            nn.Conv2d(in_channels=conv_dim*8, out_channels=1,
                        kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            )
        # Used for domain classification.
        self.domain_cls = nn.Conv2d(conv_dim * 8, label_dim, 4, 1, 0, bias=False)

    def forward(self, input):
        feat = self.feat_extractor(input)
        prob = self.prob(feat)
        dom_cls = self.domain_cls(feat)
        return prob.view(-1), dom_cls.view(dom_cls.size(0), dom_cls.size(1))
