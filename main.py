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
# import AlexNet
# import VGG

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = torch.device("cuda:0")
print(device)

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

def train():
    Batch_size = 16
    EPOCH = 100

    transform = transforms.Compose(
    [transforms.Resize(256),
    transforms.ToTensor(), 
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),]
    )

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_size,
                                            shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=Batch_size,
                                            shuffle=False, num_workers=0)


    # net = torch.load('AlexNet.pkl')
    # net = AlexNet.AlexNet()
    net = VGG.VGG('VGG19')
    print(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_data = len(trainloader)

    end = time.time()

    for epochs in range(EPOCH):
        
        net.train(True)
        batch_time_mean = []
        end = time.time()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            batch_time = time.time() - end
            end = time.time()
            batch_time_mean.append(batch_time)
            
            
            if i % (int(train_data /10)) == 0 and i > 1:
                print('[%d, %5d] loss: %f' % (epochs+1, i+1, loss.item()))
                

        
        print('batch time = %f' % (np.mean(batch_time_mean)))
        test(net, testloader)
        



    # torch.save(net, 'AlexNet.pkl')





if __name__ == "__main__":
    train()