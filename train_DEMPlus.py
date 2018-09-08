from Mymodels.DEM import DEM
#import torch.optim as optim
#from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
#from SimpleClassfier.Claafier_Data_ready import ClassfierDataset
from utils.prepare_data import *
from SimpleClassfier.Classfier import *
import time


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    batch_size = 32
    lr = 1e-4
    epochs = 50
    train_dataset = ClassfierDataset('train_data/')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    semantic = word_embeding_get()
    semantic = torch.from_numpy(semantic).cuda()
    myResnet = torch.load('./SimpleClassfier/classfier_model/classfier_notNorm.pt').res
    word_label = np.load('train_data/word_labels.npy')

    model = DEM(myResnet)
    model.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-8)

    loss_list = []
    for i in range(epochs):
        count = 0.
        sum_loss = 0.
        start_time = time.time()
        model.train()

        for idx, batch_data in enumerate(train_dataloader):
            image = batch_data['image'].float().cuda()
            label = batch_data['label'].long().cuda()
            """
            if i%500 ==0 and i>0:
                lr = lr*0.991
                adjust_learning_rate(optimizer,lr)
            """
            model.zero_grad()
            loss = model.get_loss(image,label,semantic)
            loss.backward()
            optimizer.step()
            print('current loss: ' + str(loss.data) + ' batch/sum : ' + str(idx) + '/' + str(int(train_dataset.__len__() / batch_size)))
            count += 1
        end_time = time.time()
        print("finish epoch " + str(i) + " with " + str(end_time - start_time) + "s")
        print("mean loos : " + str(sum_loss / count))
        print("finish epoch " + str(i) + ' mean loss: ' + str(loss.data))
        loss_list.append(float(loss))
        torch.save(model, 'models_file/DEM.pt')
    Test = plt.plot(loss_list, label='Train')
    plt.xlabel('Epochs')
    plt.ylabel('LOSS')
    plt.legend()
    plt.savefig("log_pics/loss.png")
    plt.clf()
