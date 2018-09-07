from DEM import DEM
import torch.optim as optim
from Data_ready import MyDataset,TestDataset
from torch.utils.data import DataLoader
from utils import *
import matplotlib.pyplot as plt
from SimpleClassfier.Classfier import Classfier
from SimpleClassfier.Claafier_Data_ready import ClassfierDataset

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    train_dataset = ClassfierDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)

    semantic = torch.from_numpy(train_dataset.semantic_features).float().cuda()
    myResnet = torch.load('./SimpleClassfier/classfier_model/classfier.pt').res
    word_label = np.load('train_data/word_labels.npy')

    model = DEM(myResnet)
    model.cuda()
    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)


    lr = 0.01
    epochs = 2500
    loss_list = []
    for i in range(epochs):
        model.train()
        for idx,batch_data in enumerate(train_dataloader):
            """
            if i%500 ==0 and i>0:
                lr = lr*0.991
                adjust_learning_rate(optimizer,lr)
            """
            model.zero_grad()
            loss = model.get_loss()
            #loss.backward()
            optimizer.step()
            #loss_list.append(float(loss))
            #print("finish epoch " + str(i) + ' current loss: ' + str(loss.data))
        torch.save(model, 'DEM.pt')
    Test = plt.plot(loss_list, label='Train')
    plt.xlabel('Epochs')
    plt.ylabel('LOSS')
    plt.legend()
    plt.savefig("log_pics/loss.png")
    plt.clf()
