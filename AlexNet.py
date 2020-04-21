import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = torch.device("cuda:1")
print(device)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            
            
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
                        
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
        # return x
    
def test(model, testloader):
    model.to(device)
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

transform = transforms.Compose(
[transforms.Resize(256),
transforms.ToTensor(), 
transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),]
)

Batch_size = 64
EPOCH = 20

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=Batch_size,
                                         shuffle=False, num_workers=0)


# net = torch.load('AlexNet.pkl')
net = AlexNet()
print(net)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

train_data = len(trainloader)

end = time.time()

for epochs in range(EPOCH):
    
    batch_time_mean = []
    end = time.time()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        batch_time = time.time() - end
        end = time.time()
        batch_time_mean.append(batch_time)
        
        
        if i % (int(train_data /10)) == 0 and i > 1:
            print('[%d, %5d] loss: %.3f' % (epochs+1, i+1, running_loss / (int(train_data /10))))
            running_loss = 0.0

    
    print('batch time = %f' % (np.mean(batch_time_mean)))
    test(net, testloader)



# torch.save(net, 'AlexNet.pkl')















