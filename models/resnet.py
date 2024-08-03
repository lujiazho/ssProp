'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager, nullcontext


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout=0.0):
        super(BasicBlock, self).__init__()

        self.dropout = dropout

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if self.dropout > 0:
            self.dropout1 = nn.Dropout(self.dropout)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if self.dropout > 0:
            self.dropout2 = nn.Dropout(self.dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            ) if self.dropout == 0 else nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                nn.Dropout(self.dropout)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        if self.dropout > 0:
            out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        if self.dropout > 0: # to make it a fair comparison with our method, we have to add dropout after the conv layer
            out = self.dropout2(out)
        
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dropout=0.0):
        super(Bottleneck, self).__init__()

        self.dropout = dropout

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if self.dropout > 0:
            self.dropout1 = nn.Dropout(self.dropout)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if self.dropout > 0:
            self.dropout2 = nn.Dropout(self.dropout)

        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        if self.dropout > 0:
            self.dropout3 = nn.Dropout(self.dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            ) if self.dropout == 0 else nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                nn.Dropout(self.dropout)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        if self.dropout > 0:
            out = self.dropout1(out)

        out = self.conv2(out)
        out = F.relu(self.bn2(out))
        if self.dropout > 0:
            out = self.dropout2(out)
                                
        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropout > 0:
            out = self.dropout3(out)
            
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, input_channel=3, image_size=32, dropout=0.0):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.image_size = image_size
        self.dropout = dropout
        
        self.scaler = 1
        if self.image_size > 32:
            self.scaler = self.image_size // 32

        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if self.dropout > 0:
            self.dropout1 = nn.Dropout(self.dropout)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.dropout))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    # @contextmanager
    # def warmup_scope(self, context=None):
    #     for m in self.modules():
    #         if hasattr(m, 'mode'):
    #             m.mode = 'normal'
    #     # if context is not None:
    #     #     print(f"{context}: Switched to warmup phase")
        
    #     try:
    #         yield None
    #     finally:
    #         for m in self.modules():
    #             if hasattr(m, 'mode'):
    #                 m.mode = 'efficient'
    #         # if context is not None:
    #         #     print(f"{context}: Restored efficient phase")

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        if self.dropout > 0:
            out = self.dropout1(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4 * self.scaler)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    # def fc_feature(self, x):
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.layer4(out)
    #     out = F.avg_pool2d(out, 4)
    #     feature = out.view(out.size(0), -1)
    #     return feature
    
    # def fc_forward(self, x):
    #     out = self.linear(x)
    #     return out

def ResNet6(num_classes=10, input_channel=3, image_size=32, dropout=0.0):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes, input_channel, image_size, dropout=dropout)

def ResNet18(num_classes=10, input_channel=3, image_size=32, dropout=0.0):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, input_channel, image_size, dropout=dropout)


def ResNet26(num_classes=10, input_channel=3, image_size=32, dropout=0.0):
    return ResNet(BasicBlock, [2, 3, 5, 2], num_classes, input_channel, image_size, dropout=dropout)


def ResNet34(num_classes=10, input_channel=3, image_size=32, dropout=0.0):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, input_channel, image_size, dropout=dropout)


def ResNet50(num_classes=10, input_channel=3, image_size=32, dropout=0.0):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, input_channel, image_size, dropout=dropout)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()