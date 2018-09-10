from Mymodels.DEM import DEM
from Mymodels.DEM_Plus import DEM_Plus
#import torch.optim as optim
#from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
#from SimpleClassfier.Claafier_Data_ready import ClassfierDataset
from utils.prepare_data import *
from SimpleClassfier.Classfier import *
import time
from Data_ready import *

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_DEMplus():
    batch_size = 16
    lr = 1e-4
    epochs = 50
    model_path = 'models_file/DEM_Plus.pt'

    train_dataset = ClassfierDataset('data/train_data/')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    semantic = word_embeding_get()
    semantic = torch.from_numpy(semantic).cuda()
    myResnet = torch.load('./SimpleClassfier/classfier_model/classfier_filtered_unNorm.pt').res
    word_label = np.load('data/train_data/word_labels.npy')
    if os.path.exists(model_path):
        model = torch.load(model_path)
        print('loaded pre_trained model')
    else:
        model = DEM_Plus(myResnet)
    model.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-8)

    loss_list = []
    min_loss = 9999
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
            loss = model.get_loss(image, label, semantic)
            sum_loss += float(loss.data)
            loss.backward()
            optimizer.step()
            print('current loss: ' + str(loss.data) + ' batch/sum : ' + str(idx) + '/' + str(int(train_dataset.__len__() / batch_size)))
            count += 1
        end_time = time.time()
        print("finish epoch " + str(i) + " with " + str(end_time - start_time) + "s")
        print("mean loos : " + str(sum_loss / count))
        if sum_loss / count < min_loss:
            print("保存模型文件至: " + model_path)
            torch.save(model, model_path)
            print("loss declined:" + str(min_loss - sum_loss / count))
            min_loss = sum_loss / count
        loss_list.append(float(loss))
    plt.xlabel('Epochs')
    plt.ylabel('LOSS')
    plt.legend()
    plt.savefig("log_pics/loss.png")
    plt.clf()


def eval_DEMPlus():
    word_labels, attributes = attribute_label()
    semantic = word_embeding_get()
    semantic = torch.from_numpy(semantic).cuda()
    eval_dataset = DEMPlusEvalDataset()
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=1)
    model = torch.load('models_file/DEM_plus.pt')
    model.eval()

    sum_len = eval_dataset.__len__()
    got_it = 0
    for idx, batch_data in enumerate(eval_dataloader):
        image = batch_data['image'].float().cuda()
        label = batch_data['label']
        pre = model.predict(image, semantic)
        if word_labels[pre[0]] == label[0]:
            got_it += 1

        print('processed image ' + str(idx))
        print('pre: '+word_labels[pre[0]] +' ground_truth: '+label[0]+' got: '+str(got_it))
    print(got_it/sum_len)

if __name__ == "__main__":
    #eval_DEMPlus()
    train_DEMplus()