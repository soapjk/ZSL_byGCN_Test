from Data_ready import TestDataset
from torch.utils.data import DataLoader
from Utils import *
from Mymodels import DEM
import torch
import numpy as np
from Utils.prepare_data import *




if __name__ == '__main__':

    test_dataset = TestDataset()
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    semantic = torch.from_numpy(test_dataset.semantic_features).float().cuda()
    model = torch.load('models_file/DEM.pt')
    model.eval()
    _, attribute = attribute_label()
    attribute = torch.from_numpy(attribute).cuda()
    for idx, batch_data in enumerate(test_dataloader):
        image = batch_data['image'].float().cuda()
        pre = model.predict(image, semantic,attribute)
        image_name = images_name[idx]
        ready_to_write_label = label_list[pre[0]]
        result_file.write(image_name+'\t'+ready_to_write_label+'\n')
        print('processed image '+str(idx)+': '+image_name+'\t'+ready_to_write_label+'\n')
        #print(all_prediction)
    pass




