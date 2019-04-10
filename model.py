import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, block, num_blocks=6):
        super(Generator, self).__init__()
        self.classname = self.__class__.__name__
        self.layers = []
        self.block = block
        self.in_planes = 64

        # Downsampling
        self.layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2))
        self.layers.append(nn.InstanceNorm2d(num_features=128))
        self.layers.append(nn.ReLU())

        self.layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2))
        self.layers.append(nn.InstanceNorm2d(num_features=256))
        self.layers.append(nn.ReLU())

        # Bottleneck
        self._make_layer(block, 256, num_blocks, stride=2)

        # Upsampling
        self.layers.append(nn.ConvTranspose2d(in_channels= 256, out_channels=128, kernel_size=3, stride=2))
        self.layers.append(nn.InstanceNorm2d(num_features=128))
        self.layers.append(nn.ReLU())

        self.layers.append(nn.ConvTranspose2d(in_channels= 128, out_channels=64, kernel_size=3 , stride=2))
        self.layers.append(nn.InstanceNorm2d(num_features=64))
        self.layers.append(nn.ReLU())

        self.model = nn.Sequential(*self.layers)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        # layers = []
        for stride in strides:
            self.layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        # return nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.classname = self.__class__.__name__
        self.layers = []

        self.layers.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1))
        self.layers.append(nn.LeakyReLU(negative_slope=0.2))

        self.layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1))
        self.layers.append(nn.LeakyReLU(negative_slope=0.2))

        self.layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1))
        self.layers.append(nn.LeakyReLU(negative_slope=0.2))

        self.layers.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1))
        self.layers.append(nn.LeakyReLU(negative_slope=0.2))

        self.layers.append(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=2, padding=1))
        self.layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*self.layers)


    def forward(self, x):
        return self.model(x)


# Xavier weight initialization
def init_weights(m):
    if m.classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

    elif m.classname.find("InstanceNorm") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
