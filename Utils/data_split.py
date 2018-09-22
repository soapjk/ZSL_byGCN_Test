from Utils.prepare_data import *

train_path = '../../Data/DatasetB_20180919/'
test_path = '../../Data/DatasetB_20180919/'
train_npy_path = '../../Data/data/B/train_data/'
test_npy_path = '../../Data/data/B/test_data/'
eval_npy_path = '../../Data/data/B/eval_data/'
path = train_path


def split_images_label_ready():
    if not os.path.exists(eval_npy_path+'B_eval_images.npy'):
        file_train = open(path + 'train.txt', 'r', encoding='utf-8')
        train_lines = file_train.readlines()
        num = 0
        image_train_list = []
        image_eval_list = []
        label_train_list = []
        label_eval_list = []
        label_num_dict = dict()
        label_count = dict()
        word_labels, attributes = attribute_label()
        for i in range(word_labels.shape[0]):
            label_num_dict[word_labels[i]] = i
            label_count[word_labels[i]] = 0

        for line in train_lines:
            print(num)
            line = line[:-1].split('\t')
            word_label = line[1]

            if num % 5 == 0:
                image_eval_list.append(cv2.imread(path+'train/'+line[0]))
                label_eval_list.append(label_num_dict[word_label])
            else:
                image_train_list.append(cv2.imread(path+'train/'+line[0]))
                label_train_list.append(label_num_dict[word_label])

            num += 1
        image_eval_list = np.array(image_eval_list)
        np.save(eval_npy_path+'B_eval_images.npy', image_eval_list)
        image_train_list = np.array(image_train_list)
        np.save(train_npy_path + 'B_train_images.npy', image_train_list)
        label_train_list = np.array(label_train_list)
        label_eval_list = np.array(label_eval_list)
        np.save(train_npy_path + 'B_train_label.npy', label_train_list)
        np.save(eval_npy_path + 'B_eval_label.npy', label_eval_list)

    else:
        image_eval_list = np.load(train_npy_path+'B_eval_images.npy')
        image_train_list = np.load(train_npy_path+'B_train_images.npy')
        label_train_list = np.load(train_npy_path + 'B_train_label.npy')
        label_eval_list = np.save(train_npy_path + 'B_eval_label.npy')

    return image_eval_list, image_train_list, label_eval_list, label_train_list


if __name__ == '__main__':
    split_images_label_ready()
