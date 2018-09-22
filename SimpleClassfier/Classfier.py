import sys
import torch
import torch.nn as nn
import torch.optim as optim
from SimpleClassfier.Claafier_Data_ready import *
from SimpleClassfier.MyResnet import MyResNet
from SimpleClassfier.MyInspection import MyInspectionV3
from SimpleClassfier.DenseNet import DenseNet

from torch.utils.data import DataLoader
import time
import os
import numpy as np
sys.path.append('../')


class ClassLoss(nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()

    def forward(self, x, label):
        #loss = torch.sum(torch.abs(x-label))
        loss = nn.CrossEntropyLoss()(x, label)
        return loss


class Classfier(nn.Module):

    def __init__(self, size):

        super(Classfier, self).__init__()
        self.cnn = MyResNet(size)
        #self.cnn = MyInspectionV3(size)
        self.size = size
        self.weight = nn.Parameter(torch.Tensor(285, self.size), requires_grad=True)
        nn.init.uniform_(self.weight, -1, 1)

    def forward(self, x):
        x = self.cnn(x)
        """
        l2_number = torch.sqrt(torch.sum(torch.pow(self.weight, 2), 1))
        l2_number = torch.unsqueeze(l2_number, 1)
        one_vector = torch.ones([1, 128]).float().cuda()
        maxtrix_l2 = torch.mm(l2_number, one_vector)
        using_weight = self.weight/maxtrix_l2
        """
        using_weight = self.weight

        x = torch.mm(x, torch.t(using_weight))
        return x


if __name__ == '__main__':
    batch_size = 16
    model_path = 'classfier_model/classfier_Norm_B_withEval_Densenet.pt'
    train_dataset = ClassfierDatasetTrain('../../Data/data/B/train_data/')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    eval_dataset = ClassfierDatasetEval('../../Data/data/B/eval_data/')
    eval_dataloader = DataLoader(eval_dataset, batch_size=256, shuffle=False, num_workers=1)
    if os.path.exists(model_path):
        model = torch.load(model_path)
        print("loaded pretrained model.")
    else:
        #model = Classfier(64)
        model = DenseNet()
    model.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_func = ClassLoss()
    epochs = 200
    best_epochs = 0
    max_score = 0
    for i in range(epochs):
        count = 0.
        sum_loss = 0.
        start_time = time.time()
        model.train()
        # train part
        """
        for idx, batch_data in enumerate(train_dataloader):
            images = batch_data['image'].float().cuda()
            label_num = batch_data['label'].long().cuda()
            #mask = batch_data['labels_one_hot'].float().cuda()
            model.zero_grad()
            pre = model(images)
            loss = loss_func(pre, label_num)
            count += 1
            sum_loss += float(loss)
            loss.backward()
            optimizer.step()
            print('current loss: '+str(loss.data)+' batch/sum : '+str(idx)+'/'+str(int(train_dataset.__len__()/batch_size)))
        """
        # eval part
        model.eval()
        right_num = 0
        len = eval_dataset.__len__()
        for idx, batch_data in enumerate(eval_dataloader):
            images = batch_data['image'].float().cuda()
            label_num = batch_data['label'].long().numpy()
            pre = model(images)
            pre = pre.detach().cpu().numpy()
            pre = np.argmax(pre, 1)
            op = pre == label_num
            right_num += np.sum(op)
            print(right_num)
        score = right_num/len
        end_time = time.time()
        print("finish epoch " + str(i) + " with " + str(end_time - start_time) + "s")
        print("current score : " + str(score))
        if score > max_score:
            best_epochs = i
            print("保存模型文件至: " + model_path)
            torch.save(model, model_path)
            print("score rised:"+str(score-max_score))
            max_score = score
    print('best epoch is : '+str(best_epochs))
    print('max score is : '+str(max_score))


