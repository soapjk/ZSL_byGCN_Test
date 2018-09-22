from Mymodels.DEM_ordinary import DEMORdinary
#import torch.optim as optim
#from torch.Utils.data import DataLoader
import matplotlib.pyplot as plt
#from SimpleClassfier.Claafier_Data_ready import ClassfierDataset
from Utils.prepare_data import *
from SimpleClassfier.Classfier import *
import time
from Data_ready import *

model_path = 'models_file/DEM_ordinary_norm_B_128.pt'
def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_dem_odinary(epochs=50):

    batch_size = 32
    train_dataset = ClassfierDatasetTrain('../Data/data/B/train_data/')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    eval_dataset = ClassfierDatasetEval('../Data/data/B/eval_data/')
    eval_dataloader = DataLoader(eval_dataset, batch_size=256, shuffle=False, num_workers=1)
    semantic = word_embeding_get()
    semantic = torch.from_numpy(semantic).cuda()
    cnn = torch.load('./SimpleClassfier/classfier_model/classfier_Norm_B_withEval_Densenet.pt')
    if os.path.exists(model_path):
        print('load modelfile')
        model = torch.load(model_path)
    else:
        model = DEMORdinary(cnn)
    model.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-8)

    max_score = 0.

    for i in range(epochs):
        count = 0.
        sum_loss = 0.
        start_time = time.time()

        # train part
        model.train()
        for idx, batch_data in enumerate(train_dataloader):
            image = batch_data['image'].float().cuda()
            label = batch_data['label'].long().cuda()
            model.zero_grad()
            loss = model.get_loss(image, label, semantic)
            loss.backward()
            optimizer.step()
            print('current loss: ' + str(loss.data) + ' batch/sum : ' + str(idx) + '/' + str(int(train_dataset.__len__() / batch_size)))
            count += 1
            sum_loss += float(loss)

        # eval part
        model.eval()
        right_num = 0
        len = eval_dataset.__len__()
        for idx, batch_data in enumerate(eval_dataloader):
            images = batch_data['image'].float().cuda()
            label_num = batch_data['label'].long().numpy()
            pre, _ = model.predict(images, semantic)
            op = pre == label_num
            right_num += np.sum(op)
            print(right_num)
        score = right_num / len
        end_time = time.time()
        print("finish epoch " + str(i) + " with " + str(end_time - start_time) + "s")
        print("current score : " + str(score))
        if score > max_score:
            best_epochs = i
            print("保存模型文件至: " + model_path)
            torch.save(model, model_path)
            print("score rised:" + str(score - max_score))
            max_score = score



def eval_DEM():
    word_labels, attributes = attribute_label()
    semantic = word_embeding_get()
    semantic = torch.from_numpy(semantic).cuda()
    eval_dataset = DEMPlusEvalDataset()
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=1)
    model = torch.load(model_path)
    model.eval()

    sum_len = eval_dataset.__len__()
    got_it = 0
    for idx, batch_data in enumerate(eval_dataloader):
        image = batch_data['image'].float().cuda()
        label = batch_data['label']
        pre = model.predict(image, semantic)
        print('processed image ' + str(idx))

        if word_labels[pre] == label[0]:
            got_it += 1
        print(pre)
        print('pre: ' + word_labels[pre] + ' ground_truth: ' + label[0] + ' got: ' + str(got_it))
        pass
    print(got_it / sum_len)


def get_result():
    # 训练文件目录，原文件结构无需改动
    test_path = '../Data/DatasetA_test_20180813/DatasetA_test/'
    result_file = open('result/submit_unorm_B_dense_cosin.txt', 'a', encoding='utf-8')
    label_list = np.load('../Data/data/B/train_data/B_label_list.npy')
    images_name = np.load('../Data/data/B/test_data/B_images_name.npy')
    test_dataset = TestDataset()
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    semantic = torch.from_numpy(test_dataset.semantic_features).float().cuda()
    model = torch.load(model_path)
    model.eval()
    _, attribute = attribute_label()
    attribute = torch.from_numpy(attribute).cuda()
    visual_emb_list = []
    for idx, batch_data in enumerate(test_dataloader):
        image = batch_data['image'].float().cuda()
        pre, visual_emb = model.predict(image, semantic)
        image_name = images_name[idx]
        ready_to_write_label = label_list[pre]
        result_file.write(image_name + '\t' + ready_to_write_label[0] + '\n')
        print('processed image ' + str(idx) + ': ' + image_name + '\t' + ready_to_write_label[0] + '\n')
        visual_emb_list.append(visual_emb)
    visual_emb_list = np.array(visual_emb_list)
    np.save('result/visual_emd_342.npy', visual_emb_list)
    # print(all_prediction)


if __name__ == "__main__":
    #train_dem_odinary(30)
    get_result()
    pass
