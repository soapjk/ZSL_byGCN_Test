import numpy as np
import cv2
train_file = open('../../Data/DatasetA_train_20180813/train.txt', 'r', encoding='utf-8')
train_lines = train_file.readlines()
eval_image_list = []
eval_label_list = []
i = 1
for line in train_lines[30000:]:
    print('processing image: '+str(i))
    i+=1
    line = line[:-1].split('\t')
    image_name = line[0]
    label = line[1]
    image = cv2.imread('../../Data/DatasetA_train_20180813/train/'+image_name)
    eval_image_list.append(image)
    eval_label_list.append(label)

eval_label_list = np.array(eval_label_list)
eval_image_list = np.array(eval_image_list)
np.save('../data/eval_data/eval_label_list.npy', eval_label_list)
np.save('../data/eval_data/eval_image_list.npy', eval_image_list)