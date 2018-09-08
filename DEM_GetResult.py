from Data_ready import TestDataset
from torch.utils.data import DataLoader
from utils import *
from Mymodels import DEM
import torch
import numpy as np

# 训练文件目录，原文件结构无需改动
test_path = '../Data/DatasetA_test_20180813/DatasetA_test/'
result_file = open('result/submit_DEM.txt', 'a', encoding='utf-8')
label_list = np.load('test_data/label_list.npy')
images_name = np.load('test_data/images_name.npy')


if __name__ == '__main__':

    test_dataset = TestDataset()
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    semantic = torch.from_numpy(test_dataset.semantic_features).float().cuda()
    model = torch.load('models_file/DEM.pt')
    model.eval()

    for idx, batch_data in enumerate(test_dataloader):
        image = batch_data['image'].float().cuda()
        pre = model.predict(image, semantic)
        image_name = images_name[idx]
        ready_to_write_label = label_list[pre[0]]
        result_file.write(image_name+'\t'+ready_to_write_label+'\n')
        print('processed image '+str(idx)+': '+image_name+'\t'+ready_to_write_label+'\n')
        #print(all_prediction)
    pass




