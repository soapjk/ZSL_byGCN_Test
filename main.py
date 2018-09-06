from GCNmodel import Mymodel
import os
import torch
import torch.nn as nn
import torch.optim as optim
from Data_ready import MyDataset,TestDataset
from torch.utils.data import DataLoader
import numpy as np
from SimpleClassfier.Classfier import Classfier
from utils import *
import matplotlib.pyplot as plt


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, graph_feature, target_weight):
        #loss = torch.sum(torch.abs(semantic_feature[0:190] - classfier_weight[0:190]))
        #loss = torch.sum(torch.pow(semantic_feature[0:190] - classfier_weight[0:190], 2))/190
        #loss = nn.MSELoss()(graph_feature[0:190], target_weight[0:190])
        loss = 0
        for i in range(190):
            loss += nn.MSELoss()(semantic_feature[i], target_weight[i])

        return loss/190


if __name__ == "__main__":
    train_dataset = MyDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    test_dataset = TestDataset()
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    myResnet = torch.load('./SimpleClassfier/classfier_model/classfier.pt').res
    myResnet.cuda()
    myResnet.eval()
    word_label = np.load('train_data/word_labels.npy')

    model = Mymodel(graph=train_dataset.graph)
    model.cuda()
    loss_function = MyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    semantic = torch.from_numpy(train_dataset.semantic_features).float().cuda()
    classfier_weight = torch.from_numpy(train_dataset.classfierweight).float().cuda()
    classfier_weight = L2_Normalize(classfier_weight)

    epochs = 30000
    loss_list = []
    for i in range(300):
        model.train()
        #for idx, batch_data in enumerate(train_dataloader):
        model.zero_grad()

        semantic_feature = model(semantic)
        loss = loss_function(semantic_feature, classfier_weight)
        #print("",loss)
        loss.backward()
        optimizer.step()
        loss_list.append(float(loss))

    Test = plt.plot(loss_list, label='Train')
    plt.xlabel('Epochs')
    plt.ylabel('LOSS')
    """
    for i in range(loss_list.__len__()):
        plt.text(i, loss_list[i], str((i, round(loss_list[i], 3))), family='serif', style='italic', ha='right', wrap=True)
    """
    plt.legend()
    plt.savefig("log_pics/loss.png")
    plt.clf()
    """
    model.eval()
    for idx, batch_data in enumerate(test_dataloader):
        image = batch_data['image'].float().cuda()
        test_image = image
        test_image_feature = myResnet(test_image)
        semantic_feature = model(semantic)
        all_prediction = torch.mm(test_image_feature, torch.t(semantic_feature))
        print(all_prediction)
    """
