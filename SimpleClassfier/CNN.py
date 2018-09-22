import sys
sys.path.append('../')
import torch.nn as nn
import torch
from Utils.utils import L2_Normalize
from Data_ready import MyDataset
from torch.utils.data import DataLoader
import cv2
import numpy as np

class ContrastNormalize(nn.Module):
    def __init__(self):
        super(ContrastNormalize, self).__init__()
        pass
    def forward(self, input):
        one = torch.ones(input.shape).type(input.type())
        mean = torch.sum(input)/torch.sum(one)
        out = input/mean
        return out


class CNN_M(nn.Module):
    def __init__(self):
        """
         input size(3*64*64)
        """
        super(CNN_M, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=6, stride=2, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            ContrastNormalize(),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            ContrastNormalize(),
            nn.ReLU()

        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(3200, 128)
        pass

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        """
        l2_number = torch.sqrt(torch.sum(torch.pow(x, 2), 1))
        l2_number = torch.unsqueeze(l2_number, 1)
        one_vector = torch.ones([1, 128]).float().cuda()
        maxtrix_l2 = torch.mm(l2_number, one_vector)
        x = x/maxtrix_l2
        """
        return x

"""
if __name__ == '__main__':
    image = cv2.imread('test.jpeg')
    image = np.transpose(image,[2,0,1])
    image = torch.Tensor(image).float().cuda()
    image = torch.unsqueeze(image,0)

    model = CNN_M()
    model.cuda()
    pre = model(image)
    pass
"""