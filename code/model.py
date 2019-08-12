import config
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
    def __init__(self, in_dim=40, conv_channels=[512, 256, 128, 64],
                        conv_dim=64, out_dim=3, n_res_block=6):
        super(Generator, self).__init__()
        layers = []
        # Residual Blocks
        for i in range(n_res_block):
            layers.append(ResidualBlock(in_dim, in_dim))
        # Up-sampling, for 1x1 input, first convTranspose will output 4x4.
        # Then each convTranspose will upsample by 2x.
        in_channels, stride, padding = in_dim, 1, 0
        for i, out_channels in enumerate(conv_channels):
            layers.extend([
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=4, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            ])
            in_channels, stride, padding = out_channels, 2, 1
        self.feat_extractor = nn.Sequential(*layers)
        # Final up-sample, by 2x.
        self.img_gen = nn.Sequential(
            nn.ConvTranspose2d(in_channels=conv_channels[-1], out_channels=out_dim,
                                kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.img_gen(self.feat_extractor(input))

class Discriminator(nn.Module):
    def __init__(self, in_dim=3, conv_channels=[64, 128, 256, 512], leaky_slope=0.2, label_dim=40):
        super(Discriminator, self).__init__()
        layers = []
        # Down-sampling, each conv will downsample by 2x.
        in_channels = in_dim
        for i, out_channels in enumerate(conv_channels):
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=4, stride=2, padding=1, bias=False))
            if i: layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(leaky_slope, inplace=True))
            in_channels = out_channels
        self.feat_extractor = nn.Sequential(*layers)
        # Used to classify whether real/fake (probability)
        self.prob = nn.Sequential(
            nn.Conv2d(in_channels=conv_channels[-1], out_channels=1,
                        kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            )
        # Used for domain classification.
        self.domain_cls = nn.Conv2d(in_channels=conv_channels[-1], out_channels=label_dim,
                            kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, input):
        feat = self.feat_extractor(input)
        prob = self.prob(feat)
        dom_cls = self.domain_cls(feat)
        return prob.view(-1), dom_cls.view(dom_cls.size(0), dom_cls.size(1))
