import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models.resnet import ResNet, BasicBlock

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Conv2d(1, 64, 3, 1, padding=1)
        self.layer2 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.layer4 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer5 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm1d(12544)
        self.layer6 = nn.Linear(12544, 512)
        self.layer7 = nn.Linear(512, 10)

    def forward(self, x):
        conv1 = F.relu(self.layer1(x))
        conv2 = F.max_pool2d(F.relu(self.layer2(conv1)), 2)
        bn1 = self.bn1(conv2)
        conv3 = F.relu(self.layer3(bn1))
        conv4 = F.max_pool2d(F.relu(self.layer4(conv3)), 2)
        bn2 = self.bn2(conv4)
        conv5 = F.max_pool2d(F.relu(self.layer5(bn2)), 2)
        fc_in = conv5.view(conv3.size(0), -1)
        bn3 = self.bn3(fc_in)
        flat1 = F.relu(self.layer6(bn3))
        flat2 = F.softmax(self.layer7(flat1))

        return flat2

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layer1 = nn.Conv2d(1, 6, 3, padding=1)
        self.layer2 = nn.Conv2d(6, 16, 5)
        self.layer3 = nn.Linear(400, 120)
        self.layer4 = nn.Linear(120, 84)
        self.layer5 = nn.Linear(84, 10)

    def forward(self, x):
        conv1 = F.max_pool2d(self.layer1(x), 2)
        conv2 = F.max_pool2d(self.layer2(conv1), 2)
        fc_in = conv2.view(conv2.size(0), -1)
        flat1 = self.layer3(fc_in)
        flat2 = self.layer4(flat1)
        flat3 = self.layer5(flat2)

        return flat3

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.layer3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.layer4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.layer5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.layer6 = nn.Linear(256*6*6, 4096)
        self.layer7 = nn.Linear(4096, 4096)
        self.layer8 = nn.Linear(4096, num_classes)

    def forward(self, x):
        conv1 = F.relu(self.layer1(x), inplace=True)
        pool1 = self.pool1(conv1)
        conv2 = F.relu(self.layer2(pool1), inplace=True)
        pool2 = self.pool2(conv2)
        conv3 = F.relu(self.layer3(pool2), inplace=True)
        conv4 = F.relu(self.layer4(conv3), inplace=True)
        conv5 = F.relu(self.layer5(conv4), inplace=True)
        pool3 = self.pool3(conv5)
        fc_in = F.dropout(pool3.view(pool3.size(0), -1))
        flat1 = F.dropout(F.relu(fc_in))
        flat2 = F.relu(flat1)
        flat3 = self.layer8(flat2)

        return flat3

class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.layer2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.layer4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.layer6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.layer7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.layer9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer14 = nn.Linear(512*7*7, 4096)
        self.layer15 = nn.Linear(4096, 4096)
        self.layer16 = nn.Linear(4096, num_classes)

    def forward(self, x):
        conv1 = F.relu(self.layer1(x))
        conv2 = F.relu(self.layer2(conv1))
        pool1 = self.pool1(conv2)
        conv3 = F.relu(self.layer3(pool1))
        conv4 = F.relu(self.layer4(conv3))
        pool2 = self.pool2(conv4)
        conv5 = F.relu(self.layer5(pool2))
        conv6 = F.relu(self.layer6(conv5))
        conv7 = F.relu(self.layer7(conv6))
        pool3 = self.pool3(conv7)
        conv8 = F.relu(self.layer8(pool3))
        conv9 = F.relu(self.layer9(conv8))
        conv10 = F.relu(self.layer10(conv9))
        pool4 = self.pool4(conv10)
        conv11 = F.relu(self.layer11(pool4))
        conv12 = F.relu(self.layer12(conv11))
        conv13 = F.relu(self.layer13(conv12))
        pool5 = self.pool5(conv13)
        fc_in = pool5.view(pool5.size(0), -1)
        flat1 = F.relu(self.layer14(fc_in))
        drop1 = F.dropout(flat1)
        flat2 = F.relu(self.layer15(drop1))
        drop2 = F.dropout(flat2)
        flat3 = self.layer16(drop2)

        return flat3

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def foward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Inception(nn.Module):
    def __init__(self, in_channels, out_channels, pool_features):
        super(Inception, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3db_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3db_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3db_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def foward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3db1 = self.branch3x3db_1(x)
        branch3x3db1 = self.branch3x3db_2(branch3x3db1)
        branch3x3db1 = self.branch3x3db_3(branch3x3db1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        output = [branch1x1, branch5x5, branch3x3db1, branch_pool]
        return torch.cat(output, 1)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class MNISTResNet(ResNet):
    def __init__(self):
        super(MNISTResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=3, bias=False)


if __name__ == "__main__":
    model = CNN()

    new_model = nn.Sequential(*list(model.children()))

    conv_model = nn.Sequential()
    for layer in model.named_modules():
        if isinstance(layer[1], nn.Conv2d):
            conv_model.add_module(layer[0], layer[1])

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.normal(m.weight.data)
            init.xavier_normal(m.weight.data)
            init.kaiming_normal(m.weight.data)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_()
