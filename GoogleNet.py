import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

class basicConv2d(nn.Module):
    def __init__(self, in_channels, out_channals, **kwargs):
        super(basicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channals, **kwargs)
        self.bn = nn.BatchNorm2d(out_channals)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class inceptionBlock(nn.Module):
    def __init__(self, _in_channels, n1x1, n3x3reduce, n3x3, n5x5reduce, n5x5, poolproj):
        super(inceptionBlock,self).__init__()

        # 1x1 conv branch
        self.b1_1x1 = basicConv2d(_in_channels, n1x1, kernel_size=1, padding=1)

        # 1x1 -> 3x3 conv branch
        self.b2_1x1 = basicConv2d(_in_channels, n3x3reduce, kernel_size=1, padding=1)
        self.b2_3x3 = basicConv2d(n3x3reduce, n3x3, kernel_size=3, padding=1)

        # 1x1 -> 5x5 conv branch
        self.b3_1x1 = basicConv2d(_in_channels, n5x5reduce, kernel_size=1, padding=1)
        self.b3_5x5 = basicConv2d(n5x5reduce, n5x5, kernel_size=5, padding=2)
        
        # 1x1 -> 3x3 conv -> 3x3 conv branch
        # self.b3_1x1 = basicConv2d(_in_channels, n5x5reduce, kernel_size=1, padding=1)
        # self.b3_5x5_1 = basicConv2d(n5x5reduce, n5x5, kernel_size=3, padding=1)
        # self.b3_5x5_2 = basicConv2d(n5x5, n5x5, kernel_size=3, padding=1)

        # max pools -> 1x1 conv branch
        self.b4_pool = nn.MaxPool2d(3,padding=1, stride=1)
        self.b4_1x1 = basicConv2d(_in_channels, poolproj, kernel_size=1, padding=1)

    def forward(self, x):
        b1 = self.b1_1x1(x)
        b2 = self.b2_3x3(self.b2_1x1(x))
        b3 = self.b3_5x5(self.b3_1x1(x))
        # b3 = self.b3_5x5_2(self.b3_5x5_1(self.b3_1x1(x)))
        b4 = self.b4_1x1(self.b4_pool(x))
        
        return torch.cat([b1, b2, b3, b4], dim=1)

class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64,192,kernel_size=3,stride=1,padding=2),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        
        # 3a -> 3b
        self.block3 = nn.Sequential(
            inceptionBlock(192, 64,96,128,16,32,32),
            inceptionBlock(256, 128,128,192,32,96,64),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        
        # 4a -> 4b -> 4c -> 4d -> 4e
        self.block4 = nn.Sequential(
            inceptionBlock(480, 192,96,208,16,48,64),
            inceptionBlock(512, 160,112,224,24,64,64),
            inceptionBlock(512, 128,128,256,24,64,64),
            inceptionBlock(512, 112,144,288,32,64,64),
            inceptionBlock(528, 256,160,320,32,128,128),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        
        # 5a -> 5b
        self.block5 = nn.Sequential(
            inceptionBlock(832, 256,160,320,32,128,128),
            inceptionBlock(832, 384,192,384,48,128,128),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        
        self.avg_pool = nn.Sequential(
            # nn.AvgPool2d(kernel_size=7, stride=1)
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.dropout = nn.Sequential(
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 10)            
        )

    def forward(self, x):  
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
if __name__ == "__main__":
    net = GoogleNet()
    print(net)
    
    # x = torch.randn(1,3,256,256)
    # print(net(x))
    # if torch.cuda.is_available():
    #     net.cuda()
    # torchsummary.summary(net, (3, 224, 224))














