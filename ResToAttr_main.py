import sys
import torch
import torch.nn as nn
import torch.optim as optim
from SimpleClassfier.Claafier_Data_ready import ClassfierDataset
from SimpleClassfier.MyResnet import MyResNet
from torch.utils.data import DataLoader
import time
import os
from Mymodels.ResToAttr import ResToAttr
from Data_ready import *

sys.path.append('../')



def train_RTA(epochs=50):

    batch_size = 32
    model_path = 'models_file/ResToAttr.pt'
    train_dataset = ClassfierDataset('../Data/data/train_data/')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    attr = torch.from_numpy(train_dataset.attr).float().cuda()

    if os.path.exists(model_path):
        model = torch.load(model_path)
        print("loaded pretrained model.")
    else:
        model = ResToAttr()
    model.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)

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
            loss = model.get_loss(images, attr[label_num])
            count += 1
            sum_loss += float(loss)
            loss.backward()
            optimizer.step()
            print('current loss: ' + str(loss.data) + ' batch/sum : ' + str(idx) + '/' + str(int(train_dataset.__len__() / batch_size)))

        end_time = time.time()
        print("finish epoch " + str(i) + " with " + str(end_time - start_time) + "s")
        print("mean loos : " + str(sum_loss / count))
        if sum_loss / count < min_loss:
            print("保存模型文件至: " + model_path)
            torch.save(model, model_path)
            print("loss declined:" + str(min_loss - sum_loss / count))
            min_loss = sum_loss / count


def eval_DEM():
    word_labels, attributes = attribute_label()
    semantic = word_embeding_get()
    semantic = torch.from_numpy(semantic).cuda()
    eval_dataset = DEMPlusEvalDataset()
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=1)
    model = torch.load('models_file/ResToAttr.pt')
    model.eval()

    sum_len = eval_dataset.__len__()
    got_it = 0
    for idx, batch_data in enumerate(eval_dataloader):
        image = batch_data['image'].float().cuda()
        label = batch_data['label']
        pre = model.predict(image, semantic)
        print('processed image ' + str(idx))

        if word_labels[pre[0]] == label[0]:
            got_it += 1
        print(pre[0])
        print('pre: ' + word_labels[pre[0]] + ' ground_truth: ' + label[0] + ' got: ' + str(got_it))
        pass
    print(got_it / sum_len)


if __name__ == '__main__':
    train_RTA(10)
