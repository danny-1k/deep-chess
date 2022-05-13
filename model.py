import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=12,out_channels=24,kernel_size=4,stride=2,padding=1)
        self.conv2 = nn.Conv2d(in_channels=24,out_channels=48,kernel_size=4,stride=2,padding=1)
        self.fc1 = nn.Linear(192,50)
        self.fc2 = nn.Linear(50,1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    
    def forward(self,x):
        #x -> (n, 8,8,12)
        x = self.conv1(x) #(8-4 +2)/2 + 1 -> (4,4)
        x = self.relu(x)

        x = self.conv2(x) #(4-4 + 2)/2 +1 -> (2,2)
        x = self.relu(x)

        x = x.view(x.shape[0],-1)

        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

    def save_model(self):
        torch.save(self,f'checkpoints/models/Net.pt')

    def save_checkpoint(self):
        torch.save(self.state_dict(),f'checkpoints/state/Net_state_dict.pt')

    def load_checkpoint(self):
        self.load_state_dict(torch.load(f'checkpoints/state/Net_state_dict.pt'))