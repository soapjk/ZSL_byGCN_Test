import sys
sys.path.append('../')

from torch.utils.data import Dataset
import numpy as np

class ClassfierDataset(Dataset):
    def __init__(self, path):
        images = np.load(path+'images.npy')
        images = np.transpose(images, (0, 3, 1, 2))
        labels = np.load(path+'label_num.npy').astype(np.long)
        labels_one_hot = np.load(path+'labels_one_hot.npy')
        self.images = images[0:30000]
        self.labels = labels[0:30000]
        self.labels_one_hot = labels_one_hot
        pass

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        boop = {'image': image, 'label': label, 'labels_one_hot': self.labels_one_hot[index]}
        return boop

    def __len__(self):
        return len(self.images)