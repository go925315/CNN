import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class basicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channals, stride=1, downsampling=False):
        super(basicBlock, self).__init__()
        
        self.downsampling = downsampling
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channals, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channals),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channals, out_channals, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channals)
        )
        
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channals, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channals)
            )
        
    def forward(self, x):
        residual = x
        
        out = self.conv(x)
        if self.downsampling:
            residual = self.downsample(x)
            
        out += residual
        out = F.relu(out)
        return out
        
     
class bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channals, stride=1, downsampling=False):
        super(bottleneck, self).__init__()
        
        self.downsampling = downsampling
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channals, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channals),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channals, out_channals, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channals),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channals, out_channals*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channals*self.expansion)
        )
        
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channals*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channals*self.expansion)
            )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.conv(x)
        if self.downsampling:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
        )
        
        self.inplanes = 64
        
        self.conv2 = self._make_layer(block, layers[0], 64, stride=1)
        self.conv3 = self._make_layer(block, layers[1], 128, stride=2)
        self.conv4 = self._make_layer(block, layers[2], 256, stride=2)
        self.conv5 = self._make_layer(block, layers[3], 512, stride=2)
        
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion, num_classes),
            nn.Sigmoid()            
        )
        
        
    def _make_layer(self, block, layer, planes, stride=1):
        layers = []
        layers.append(block(in_channels=self.inplanes, out_channals=planes, stride=stride, downsampling =True))
        
        self.inplanes = planes * block.expansion
        for _ in range(1, layer):
            layers.append(block(in_channels=self.inplanes, out_channals=planes, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
        


def ResNet18():
    return ResNet(basicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(basicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(bottleneck, [3,8,36,3])

if __name__ == "__main__":

    net = ResNet152()
    # print(net.fc[0])
    x = torch.randn(1,3,224,224)
    print(net(x))



