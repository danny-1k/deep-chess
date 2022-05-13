import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from model import Net
import matplotlib.pyplot as plt

from tqdm import tqdm

from data import chessData


device = 'cuda' if torch.cuda.is_available() else 'cpu'


net = Net()
net.to(device)
net.save_model()


epochs = 100
lr = 1e-5

lossfn = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(),lr=lr,)

trainloader = DataLoader(chessData(train=True),batch_size=64,shuffle=True)
testloader = DataLoader(chessData(train=False),batch_size=64,shuffle=True)

train_loss_over_time = []
test_loss_over_time = []

lowest_loss = float('inf')

print('training started')

for epoch in tqdm(range(epochs)):
    
    train_loss_epoch = []
    test_loss_epoch = []

    net.train()

    for x,y in trainloader:

        x.to(device)
        y.to(device)

        p = net(x)
        loss = lossfn(p,y)

        train_loss_epoch.append(loss.item())

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

    net.eval()

    with torch.no_grad():

        x.to(device)
        y.to(device)

        for x,y in testloader:
            
            p = net(x)
            loss = lossfn(p,y)

            test_loss_epoch.append(loss.item())

    train_loss_over_time.append(sum(train_loss_epoch)/len(train_loss_epoch))
    test_loss_over_time.append(sum(test_loss_epoch)/len(test_loss_epoch))

    if test_loss_over_time[-1] < lowest_loss:
        net.save_checkpoint()
        lowest_loss = test_loss_over_time[-1]

    plt.plot(train_loss_over_time,label='Train loss over time')
    plt.plot(test_loss_over_time,label='Test loss over time')
    plt.legend()
    plt.savefig('plots/Net_Loss')
    plt.close('all')


