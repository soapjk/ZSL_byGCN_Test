import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim
from SimpleClassfier.Claafier_Data_ready import ClassfierDataset
from SimpleClassfier.MyResnet import MyResNet
from SimpleClassfier.CNN import CNN_M
from torch.utils.data import DataLoader
import time
import os
from utils import L2_Normalize

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
        self.res = MyResNet()
        #self.res = CNN_M()
        self.size = size
        self.weight = nn.Parameter(torch.Tensor(230, self.size), requires_grad=True)
        nn.init.uniform_(self.weight, -1, 1)

    def forward(self, x):
        x = self.res(x)

        l2_number = torch.sqrt(torch.sum(torch.pow(self.weight, 2), 1))
        l2_number = torch.unsqueeze(l2_number, 1)
        one_vector = torch.ones([1, 128]).float().cuda()
        maxtrix_l2 = torch.mm(l2_number, one_vector)
        using_weight = self.weight/maxtrix_l2

        #using_weight = self.weight

        x = torch.mm(x, torch.t(using_weight))
        return x


if __name__ == '__main__':
    batch_size = 1
    model_path = './classfier_model/classfier.pt'
    train_dataset = ClassfierDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    if os.path.exists(model_path):
        model = torch.load(model_path)
        print("loaded pretrained model.")
    else:
        model = Classfier(128)
    model.cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    loss_func = ClassLoss()
    epochs = 500

    min_loss = 999
    for i in range(epochs):
        count = 0.
        sum_loss = 0.
        start_time = time.time()
        model.train()

        for idx, batch_data in enumerate(train_dataloader):
            images = batch_data['image'].float().cuda()
            label_num = batch_data['label'].long().cuda()
            mask = batch_data['labels_one_hot'].float().cuda()
            model.zero_grad()
            pre = model(images)
            loss = loss_func(pre, label_num)
            count += 1
            sum_loss += float(loss)
            loss.backward()
            optimizer.step()
            print('current loss: '+str(loss.data)+' batch/sum : '+str(idx)+'/'+str(int(train_dataset.__len__()/batch_size)))

        end_time = time.time()
        print("finish epoch " + str(i) + " with " + str(end_time - start_time) + "s")
        print("mean loos : " + str(sum_loss / count))
        if sum_loss / count < min_loss:
            print("保存模型文件至: " + model_path)
            torch.save(model, model_path)
            print("loss declined:"+str(min_loss-sum_loss/count))
            min_loss = sum_loss / count

