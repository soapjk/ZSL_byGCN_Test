from torch.utils.data import Dataset
from Utils.prepare_data import *

class MyDataset(Dataset):
    def __init__(self):
        graph = graph_ready()
        semantic_features = word_embeding_get()
        self.classfierweight = np.load('data/train_data/classfierweight.npy')
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
        self.images = np.load('../Data/data/B/test_data/B_images.npy')
        self.images = np.transpose(self.images, (0, 3, 1, 2))
        semantic_features = word_embeding_get()
        self.semantic_features = semantic_features
    def __getitem__(self, index):
        image = self.images[index]
        return {'image': image}

    def __len__(self):
        return self.images.shape[0]


class DEMPlusTrainDataset(Dataset):
    def __init__(self, path):
        images = np.load(path+'images.npy')
        images = np.transpose(images, (0, 3, 1, 2))
        labels = np.load(path+'label_num.npy').astype(np.long)
        labels_one_hot = np.load(path+'labels_one_hot.npy')
        self.images = images[0:30000]
        self.labels = labels[0:30000]
        self.labels_one_hot = labels_one_hot[0:30000]

    def __getitem__(self, index):

        boop = {'image': self.images[index], 'label': self.labels[index], 'labels_one_hot': self.labels_one_hot[index]}
        return boop

    def __len__(self):
        return len(self.images)


class DEMPlusEvalDataset(Dataset):
    def __init__(self):
        images = np.load('../Data/data/eval_data/eval_image_list.npy')
        images = np.transpose(images, (0, 3, 1, 2))
        labels = np.load('../Data/data/eval_data/eval_label_list.npy')
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        boop = {'image': self.images[index], 'label': self.labels[index]}
        return boop

    def __len__(self):
        return len(self.images)