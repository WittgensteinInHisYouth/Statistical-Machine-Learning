import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

labels = {
0 : "T-shirt",
1 : "Trouser",
2 : "Pullover",
3 : "Dress",
4 : "Coat",
5 : "Sandal",
6 : "Shirt",
7 : "Sneaker",
8 : "Bag",
9 : "Ankle boot"
}


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # initialize layers here
        self.conv1 = nn.Conv2d(1,32,5,1)
        self.conv2 = nn.Conv2d(32,64,5,1)

        self.fc1 = nn.Linear(64*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=10)
    def forward(self, x):
        # invoke the layers here
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)

        x = x.view(-1, 64*4*4)

        x =F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x,dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Fill in here
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Epoch:', epoch, ',loss:', loss.item())


def  test(model,  device,  test_loader):
    model.eval()

    correct = 0
    exampleSet = False
    example_data = numpy.zeros([10,  28,  28])
    example_pred = numpy.zeros(10)
    with  torch.no_grad():
        for  data,  target  in  test_loader:
            data,  target  =  data.to(device),  target.to(device)
            #  fill  in  here
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        if  not  exampleSet:
            for  i  in  range(10):
                example_data[i] = data[i][0].to("cpu").numpy()
                example_pred[i] = pred[i].to("cpu").numpy()
            exampleSet  =  True

    print('Test  set  accuracy: ',100.  *  correct  /  len(test_loader.dataset),  '%')

    for i  in  range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(example_data[i],  cmap='gray',  interpolation='none')
        plt.title(labels[example_pred[i]])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def main():
    N_EPOCH = 5# Complete here
    L_RATE = 0.01 # Complete here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = datasets.FashionMNIST('../data', train=True,
    download=True, transform=transforms.ToTensor())
    test_dataset = datasets.FashionMNIST('../data', train=False,
    download=True, transform=transforms.ToTensor())
    ##### Use dataloader to load the datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)# Complete here
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)# Complete here
    model = CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=L_RATE)
    for epoch in range(1, N_EPOCH + 1):
        test(model, device, test_loader)
        train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)


if __name__ == '__main__':
    main()
