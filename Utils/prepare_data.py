import torch
import numpy as np
import os
import cv2
import pickle
train_path = '../../Data/DatasetB_20180919/'
test_path = '../../Data/DatasetB_20180919/'
train_A_path = '../../Data/DatasetA_train_20180813/'
train_npy_path = '../../Data/data/B/train_data/'
test_npy_path = '../../Data/data/B/test_data/'
path = train_path


def test_data_ready():
    if not os.path.exists(test_npy_path+'B_images.npy'):
        file_train = open(train_path + 'image.txt', 'r', encoding='utf-8')
        test_lines = file_train.readlines()
        num = 0
        image_list = []
        image_name_list = []
        for line in test_lines:
            print(num)
            num += 1
            line = line[:-1]
            image_list.append(cv2.imread(path + 'test/'+line))
            image_name_list.append(line)
        image_list = np.array(image_list)
        image_name_list = np.array(image_name_list)
        np.save(test_npy_path+'B_images.npy', image_list)
        np.save(test_npy_path+'B_images_name.npy', image_name_list)

    else:
        image_list = np.load(test_npy_path + 'B_images.npy')
        image_name_list = np.load(test_npy_path + 'B_images_name.npy')

    return image_list,image_name_list



def init():
    path = train_path


def attribute_label():
    if not os.path.exists(train_npy_path + 'attributes.npy') or not os.path.exists(train_npy_path + 'word_labels.npy'):
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
        np.save(train_npy_path + 'word_labels.npy', word_labels)
        np.save(train_npy_path + 'attributes.npy', attributes)
    else:
        word_labels = np.load(train_npy_path + 'word_labels.npy')
        attributes = np.load(train_npy_path + 'attributes.npy')

    return word_labels, attributes


def graph_ready():
    if not os.path.exists(train_npy_path + 'graph.npy'):
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
        graph = np.load(train_npy_path + 'graph.npy')

    return graph


def images_ready():
    if not os.path.exists(train_npy_path+'B_images.npy'):
        file_train = open(path + 'train.txt', 'r', encoding='utf-8')
        train_lines = file_train.readlines()
        num = 0
        image_list = []
        for line in train_lines:
            print(num)
            num += 1
            line = line[:-1].split('\t')
            image_list.append(cv2.imread(path+'train/'+line[0]))
        image_list = np.array(image_list)
        np.save(train_npy_path+'B_images.npy',image_list)

    else:
        image_list = np.load(train_npy_path+'B_images.npy')
    return image_list


def labels_ready():
    if not os.path.exists(train_npy_path + 'A_label_list.npy'):
        file_train = open(train_A_path + 'train.txt', 'r', encoding='utf-8')
        train_lines = file_train.readlines()
        label_list = []
        label_num_dict = dict()
        label_count = dict()
        word_labels, attributes = attribute_label()
        for i in range(word_labels.shape[0]):
            label_num_dict[word_labels[i]] = i
            label_count[word_labels[i]] = 0

        for i in range(len(train_lines)):
            print(i)
            line = train_lines[i]
            line = line[:-1].split('\t')
            label_list.append(label_num_dict[line[1]])
        label_list = np.array(label_list)
        np.save(train_npy_path + 'A_label_list.npy', label_list)
    else:
        label_list = np.load(train_npy_path + 'label_list.npy')

    return label_list


def word_embeding_get():
    if not os.path.exists(train_npy_path + 'semantic_features.npy'):
        word_featrue_file = open(path + 'class_wordembeddings.txt', 'r', encoding='utf-8')
        word_label_file = open(path + 'label_list.txt', 'r', encoding='utf-8')
        word_label_lines = word_label_file.readlines()
        word_featrue_lines = word_featrue_file.readlines()

        word_labels, attributes = attribute_label()
        semantic_features = np.zeros([word_labels.shape[0], 300], dtype=np.float32)
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
        np.save(train_npy_path + 'semantic_features.npy', semantic_features)
    else:
        semantic_features = np.load(train_npy_path + 'semantic_features.npy')

    return semantic_features


def label_list_ready():
    label_list_file = open(path + 'attributes_per_class.txt', 'r', encoding='utf-8')
    label_list_lines = label_list_file.readlines()
    label_list = []
    for line in label_list_lines:
        line = line[:-1].split('\t')
        label_list.append(line[0])
    label_list = np.array(label_list)
    np.save(train_npy_path + 'B_label_list.npy', label_list)
    np.save(test_npy_path + 'B_label_list.npy', label_list)
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
    labels_ready()
    pass

