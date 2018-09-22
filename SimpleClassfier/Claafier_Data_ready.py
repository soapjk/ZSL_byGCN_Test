import sys
sys.path.append('../')

from torch.utils.data import Dataset
import numpy as np
import cv2
class ClassfierDatasetTrain(Dataset):
    def __init__(self, path):
        images_B = np.load(path+'B_train_images.npy')
        images_A = np.load(path+'A_train_images.npy')
        images = np.concatenate((images_A, images_B))
        images = np.transpose(images, (0, 3, 1, 2))

        labels_B = np.load(path+'B_train_label.npy').astype(np.long)
        labels_A = np.load(path+'A_train_label.npy').astype(np.long)
        labels= np.concatenate((labels_A, labels_B))
        labels_hot = np.eye(285)[labels]

        self.images = images
        self.labels = labels
        self.labels_hot = labels_hot
        #self.labels_one_hot = labels_one_hot
        pass

    def __getitem__(self, index):
        image = self.images[index]

        label = self.labels[index]
        label_hot = self.labels_hot[index]
        boop = {'image': image, 'label': label,'label_hot':label_hot}
        return boop

    def __len__(self):
        return self.images.shape[0]


class ClassfierDatasetEval(Dataset):
    def __init__(self, path):
        images = np.load(path+'B_eval_images.npy')

        images = np.transpose(images, (0, 3, 1, 2))
        labels = np.load(path+'B_eval_label.npy').astype(np.long)

        labels_hot = np.eye(285)[labels]

        self.images = images
        self.labels = labels
        self.labels_hot = labels_hot

        #self.labels_one_hot = labels_one_hot
        pass

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        label_hot = self.labels_hot[index]

        boop = {'image': image, 'label': label,'label_hot':label_hot}

        return boop

    def __len__(self):
        return len(self.images)