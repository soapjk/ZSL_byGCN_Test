import torch
import numpy as np
from Utils import prepare_data


def L2_Normalize(x):

    L2_Number = torch.sqrt(torch.sum(torch.pow(x, 2), 1))
    L2_Number = torch.unsqueeze(L2_Number, 1)
    one = torch.ones([1, x.shape[1]]).type(x.type())
    l2_matrix = torch.mm(L2_Number, one)
    return x / l2_matrix


def unseen_filter():
    train_file = open('../../Data/DatasetA_train_20180813/train.txt',encoding='utf-8')
    train_lines = train_file.readlines()
    word_labels, attributes = prepare_data.attribute_label()
    mask = np.zeros(230)
    label_num_dict = dict()
    for i in range(word_labels.shape[0]):
        label_num_dict[word_labels[i]] = i

    for line in train_lines:
        line = line[:-1].split('\t')
        num = label_num_dict[line[1]]
        mask[num] = 999
    np.save('eval_unseen_mask.npy',mask)


def KNN():
    KNN_result = open('../result/submit_unorm_B_dense_cosin_KNN_10.txt','a',encoding='utf-8')
    result_path = '../result/submit_unorm_B_dense_cosin.txt'
    visual_emd = np.load('../result/visual_emd_342.npy')
    visual_emd=np.squeeze(visual_emd)
    result_file = open(result_path, 'r', encoding='utf-8')
    result_lines = result_file.readlines()
    image_name_list = []
    label_list = []
    label_num_dict = dict()
    word_labels, attributes = prepare_data.attribute_label()
    for i in range(word_labels.shape[0]):
        label_num_dict[word_labels[i]] = i

    for i in range(len(result_lines)):
        line = result_lines[i]
        line = line[:-1].split('\t')
        image_name_list.append(line[0])
        label_list.append(line[1])

    print('计算余弦距离')
    op = np.linalg.norm(visual_emd, axis=1, keepdims=True)
    mod = np.matmul(np.linalg.norm(visual_emd, axis=1,keepdims=True), op.T)
    dot_re = np.matmul(visual_emd, visual_emd.T)
    cosin_dis_matrix = dot_re/mod
    sorted_index = np.argsort(cosin_dis_matrix, axis=1)
    max_N = sorted_index[:, :10]
    print('KNN start')
    for i in range(max_N.shape[0]):
        count_n = np.zeros(285)
        for j in range(max_N.shape[1]):
            loc = max_N[i][j]
            count_n[label_num_dict[label_list[loc]]] += 1
        print(i)
        max_loc = count_n.argmax()
        #label_list[i] = word_labels[max_loc]
        ready_to_write = image_name_list[i]+'\t'+word_labels[max_loc]+'\n'
        KNN_result.write(ready_to_write)

KNN()

