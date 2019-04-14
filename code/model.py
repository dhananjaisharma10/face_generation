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

class Generator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, c_dim=40, num_blocks=6, repeat_num=2):
        super(Generator, self).__init__()
        self.classname = self.__class__.__name__
        self.layers = []
        self.block = Bottleneck
        self.image_size = image_size

        # Input dim = c_dim, Output dim = 64
        #self.layers.append(Print())
        self.layers.append(nn.Conv2d(in_channels=c_dim, out_channels=conv_dim,
                                    kernel_size=7, stride=1, padding=3, bias=False))
        #self.layers.append(Print())
        self.layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        self.layers.append(nn.ReLU())

        curr_dim = conv_dim
        # Downsampling
        for i in range(repeat_num):
            self.layers.append(nn.Conv2d(in_channels=curr_dim, out_channels=curr_dim * 2,
                                        kernel_size=4, stride=2, padding=1, bias=False))
            #self.layers.append(Print())
            self.layers.append(nn.InstanceNorm2d(num_features=curr_dim * 2, affine=True, track_running_stats=True))
            self.layers.append(nn.ReLU(inplace=True))
            curr_dim *= 2

        # Bottleneck
        #curr_dim = self._make_layer(self.block, curr_dim, curr_dim, num_blocks, stride=2)
        # Bottleneck layers.
        for i in range(num_blocks):
            self.layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Upsampling
        for i in range(repeat_num):
            self.layers.append(nn.ConvTranspose2d(in_channels=curr_dim, out_channels=curr_dim // 2,
                                                kernel_size=4, stride=2, padding=1, bias=False))
            #self.layers.append(Print())
            self.layers.append(nn.InstanceNorm2d(num_features=curr_dim // 2, affine=True, track_running_stats=True))
            self.layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        # Input dim = 64, Output dim = 3
        self.layers.append(nn.Conv2d(in_channels=curr_dim, out_channels=3,
                                    kernel_size=7, stride=1, padding=3, bias=False))
        #self.layers.append(Print())
        self.layers.append(nn.Tanh())

        self.model = nn.Sequential(*self.layers)
        self.init_weights()

    def _make_layer(self, block, in_planes, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        for stride in strides:
            self.layers.append(block(in_planes, planes, stride))
            #self.layers.append(Print())
            in_planes = planes * block.expansion
        return in_planes

    def forward(self, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, self.image_size, self.image_size)
        # x = torch.cat([x, c], dim=1)
        return self.model(c)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


class Discriminator(nn.Module):
    def __init__(self, image_size=128, slope=0.2, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=3, out_channels=conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(negative_slope=slope))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(in_channels=curr_dim, out_channels=curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(negative_slope=slope))
            curr_dim = curr_dim * 2

        layers.append(nn.Conv2d(in_channels=curr_dim, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Sigmoid()) # Optional. Used in some papers for PatchGAN discriminator

        self.model = nn.Sequential(*layers)
        self.init_weights()

        # For Domain classification loss
        # kernel_size = int(image_size / np.power(2, repeat_num))
        # self.conv2 = nn.Conv2d(in_channels=curr_dim, out_channels=c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        return self.model(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


# Initialize the Generator
# def Tiny_Generator():
#     x = Generator(Bottleneck)
#     return x


# Initialize the Discriminator
# def Tiny_Discriminator(channels=channels, slope=0.2):
#     x = Discriminator(channels, slope)
#     return x
