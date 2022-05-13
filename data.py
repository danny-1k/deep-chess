import torch
from torch.utils.data import Dataset
import random
import os
import numpy as np

class chessData(Dataset):

    def __init__(self,train=True):
        x = [os.path.join('data/x',f) for f in os.listdir('data/x')]
        y = [os.path.join('data/y',f) for f in os.listdir('data/y')]

        self.files = list(zip(x,y))
        self.train = train

        random.shuffle(self.files)

        num_train = int(.8*len(self.files))

        if self.train:
            self.files = self.files[:num_train]
        else:
            self.files = self.files[num_train:]


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        x = self.files[index][0]
        y = self.files[index][1]

        x = np.load(x,allow_pickle=True)
        y = np.load(y,allow_pickle=True)

        x = torch.from_numpy(x).permute(2,0,1).float()
        y = torch.from_numpy(y).float()

        return x,y

