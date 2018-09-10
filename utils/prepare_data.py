import torch
import numpy as np
import os
import cv2
import pickle
train_path = '../Data/DatasetA_train_20180813/'
test_path = '../Data/DatasetA_train_20180813/'
path = ''


def init():
    path = train_path


def attribute_label():
    if not os.path.exists('data/train_data/attributes.npy') or not os.path.exists('data/train_data/word_labels.npy'):
        file_attribute = open(path + 'attributes_per_class.txt', 'r', encoding='utf-8')
        attribute_lines = file_attribute.readlines()
        word_label_list = []
        attribute_list = []
        for line in attribute_lines:
            line = line[:-1].split('\t')
            word_label = line[0]
            word_label_list.append(word_label)
            attr = np.array(line[1:], dtype=np.float32)
            attribute_list.append(attr)

        word_labels = np.array(word_label_list)
        attributes = np.array(attribute_list)
        np.save('data/train_data/word_labels.npy', word_labels)
        np.save('data/train_data/attributes.npy', attributes)
    else:
        word_labels = np.load('data/train_data/word_labels.npy')
        attributes = np.load('data/train_data/attributes.npy')

    return word_labels, attributes


def graph_ready():
    if not os.path.exists('train_data/graph.npy'):
        graph = np.zeros([230, 230], dtype=np.float32)
        word_labels, attributes = attribute_label()
        #word_embeding = word_embeding_get()

        """
        for i in range(word_embeding.shape[0]):
            for j in range(word_embeding.shape[0]):
                releation = np.dot(word_embeding[i], word_embeding[j])
                #releation = releation/(np.sqrt(np.sum(np.square(word_embeding[i]))*np.sum(np.square(word_embeding[j]))))
                if releation > 0.5:
                    graph[i][j] = 1.
                    graph[j][i] = 1.
        """

        for i in range(attributes.shape[0]):
            for j in range(i+1, attributes.shape[0]):
                if np.dot(attributes[i][0:30], attributes[j][0:30]) > 0:
                    graph[i][j] = 1.0
                    graph[j][i] = 1.0

        diag_of_D = np.sum(graph, 0)
        D = np.diag(diag_of_D)
        D = np.linalg.cholesky(D)
        D = np.linalg.inv(D)
        graph = graph + np.eye(230, dtype=np.float32)
        graph = np.matmul(D, graph)
        graph = np.matmul(graph, D)
        np.save('train_data/graph.npy', graph)
    else:
        graph = np.load('train_data/graph.npy')

    return graph


def images_ready():
    if not os.path.exists('train_data/images.txt'):
        file_train = open(path + 'train.txt', 'r', encoding='utf-8')
        train_lines = file_train.readlines()

        image_list = []
        label_num_dict = dict()
        label_count = dict()
        word_labels, attributes = attribute_label()
        for i in range(word_labels.shape[0]):
            label_num_dict[word_labels[i]] = i
            label_count[word_labels[i]] = []
        mymax = 0
        for i in range(len(train_lines)):
            print('reading images:'+str(i))
            line = train_lines[i]
            line = line[:-1].split('\t')
            image = cv2.imread(path+'train/'+line[0])
            image = image.transpose(2,0,1)
            label_count[line[1]].append(image)
            if len(label_count[line[1]]) > mymax:
                mymax = len(label_count[line[1]])

        for j in range(mymax):
            this_image_list = []
            for i in range(word_labels.shape[0]):
                if len(label_count[word_labels[i]]) > 0:
                    temp = label_count[word_labels[i]].pop()
                    this_image_list.append(temp)
            this_image_list = np.array(this_image_list)
            image_list.append(this_image_list)

        f = open('train_data/images.txt', 'wb')
        pickle.dump(image_list, f, 0)
    else:
        f = open('train_data/images.txt', 'rb')
        image_list = pickle.load(f)

    return image_list

def labels_ready():
    if not os.path.exists('train_data/labels.txt'):
        file_train = open(path + 'train.txt', 'r', encoding='utf-8')
        train_lines = file_train.readlines()
        label_list = []
        label_num_dict = dict()
        label_count = dict()
        word_labels, attributes = attribute_label()
        for i in range(word_labels.shape[0]):
            label_num_dict[word_labels[i]] = i
            label_count[word_labels[i]] = 0
        mymax = 0
        for i in range(len(train_lines)):
            line = train_lines[i]
            line = line[:-1].split('\t')
            label_count[line[1]] += 1
            if label_count[line[1]] > mymax:
                mymax = label_count[line[1]]

        for j in range(mymax):
            this_label = []
            for i in range(word_labels.shape[0]):
                if label_count[word_labels[i]] > 0:
                    temp = np.zeros(230, dtype=np.float32)
                    temp[label_num_dict[word_labels[i]]] = 1.
                    this_label.append(temp)
                    label_count[word_labels[i]] -= 1
            op = np.array(this_label,dtype=np.float32)
            label_list.append(op)
        f = open('train_data/labels.txt', 'wb')
        pickle.dump(label_list, f, 0)
    else:
        f = open('train_data/labels.txt', 'rb')
        label_list = pickle.load(f)

    return label_list

def word_embeding_get():
    if not os.path.exists('data/train_data/semantic_features.npy'):
        word_featrue_file = open(path + 'class_wordembeddings.txt', 'r', encoding='utf-8')
        word_label_file = open(path + 'label_list.txt', 'r', encoding='utf-8')
        word_label_lines = word_label_file.readlines()
        word_featrue_lines = word_featrue_file.readlines()
        semantic_features = np.zeros([230,300], dtype=np.float32)
        word_labels, attributes = attribute_label()
        word_label_dict = dict()
        label_num_dict = dict()
        for i in range(word_labels.shape[0]):
            label_num_dict[word_labels[i]] = i

        for line in word_label_lines:
            line = line[:-1].split('\t')
            word_label_dict[line[1]] = line[0]

        for line in word_featrue_lines:
            line = line[:-1].split(' ')
            class_entity = line[0]
            one_semantic_feature = np.array(line[1:])
            num = label_num_dict[word_label_dict[class_entity]]
            semantic_features[num] = one_semantic_feature
        np.save('data/train_data/semantic_features.npy',semantic_features)
    else:
        semantic_features = np.load('data/train_data/semantic_features.npy')

    return semantic_features


def label_list_ready():
    label_list_file = open(path + 'label_list.txt', 'r', encoding='utf-8')
    label_list_lines = label_list_file.readlines()
    label_list = []
    for line in label_list_lines:
        line = line[:-1].split('\t')
        label_list.append(line[0])
    label_list = np.array(label_list)
    np.save('train_data/label_list.npy', label_list)
    np.save('test_data/label_list.npy', label_list)
    return label_list



def data_ready(mode='train'):
    global path
    if mode == 'train':

        path= train_path
    elif mode == 'test':
        path = test_path
    else:
        raise ValueError("Valid modes: 'test', 'train'.")
    labels = labels_ready()
    images = images_ready()
    semantic_features = word_embeding_get()
    graph = graph_ready()

    return images, graph, semantic_features, labels

def extract_classfierweight():
    model = torch.load('SimpleClassfier/classfier_model/classfier.pt')
    model.cpu()
    weight = model.weight.detach().numpy()
    np.save('train_data/classfierweight.npy', weight)

if __name__ == '__main__':
    path = test_path
    label_list_ready()
    pass

