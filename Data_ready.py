from torch.utils.data import Dataset
import numpy as np
from prepare_data import data_ready
from prepare_data import *
class MyDataset(Dataset):
    def __init__(self):
        graph = graph_ready()
        semantic_features = word_embeding_get()
        self.classfierweight = np.load('train_data/classfierweight.npy')
        self.semantic_features = semantic_features
        self.graph = graph
        #self.graph = np.eye(230,dtype=np.float32)
        pass

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        boop = {'image': image, 'label': label}

        return boop

    def __len__(self):
        return len(self.images)


class TestDataset(Dataset):
    def __init__(self):
        self.images = np.load('test_data/test_images.npy')
        self.images = np.transpose(self.images, (0, 3, 1, 2))[0:1]
        semantic_features = word_embeding_get()
        self.semantic_features = semantic_features
    def __getitem__(self, index):
        image = self.images[index]
        return {'image': image}

    def __len__(self):
        return self.images.shape[0]