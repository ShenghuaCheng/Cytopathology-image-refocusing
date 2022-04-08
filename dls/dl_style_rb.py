import os
import numpy as np
import random
import cv2
import multiprocessing
from multiprocessing import dummy, cpu_count

class DataLoader:
    def __init__(self, conf):
        self.style_3D = conf.style_3D
        self.style_our = conf.style_our
        self.data_path = conf.data_path
        self.txt_path = conf.txt_path
        self.num_per_epoch = conf.nums_per_epoch

        #     get list
        self.__get_list__()

    def __get_lines_from_txt(self, style, mode='train'):
        tmp_list = list()
        with open('%s/%s%s.txt' % (self.txt_path, style, mode), 'r') as tmp:
            for line in tmp:
                line = line.strip()
                tmp_list.append(line)
        return tmp_list

    def __get_list__(self):

        self.list_3D = self.__get_lines_from_txt(self.style_3D, '')
        self.list_our = self.__get_lines_from_txt(self.style_our, '')

        random.shuffle(self.list_3D)
        random.shuffle(self.list_our)
        print('3D data (total): %d, our data (total): %d' % (len(self.list_3D), len(self.list_our)))
        self.list_3D_train = self.list_3D[: self.num_per_epoch]
        self.list_our_train = self.list_our[: self.num_per_epoch]
        print('3D data (train): %d, our data (train): %d' % (len(self.list_3D_train), len(self.list_our_train)))

    def load_batch_train(self, batch_size=1):
        self.n_batches = int(self.num_per_epoch / batch_size)

        for i in range(self.n_batches - 1):
            batch_3D = self.list_3D_train[i * batch_size: (i + 1) * batch_size]
            batch_our = self.list_our_train[i * batch_size: (i + 1) * batch_size]
            yield self.load_batch(batch_3D, batch_our)

    def load_random_images_pair(self, batch_size=1):
        batch_3D = np.random.choice(self.list_3D, batch_size, replace=False)
        batch_our = np.random.choice(self.list_our, batch_size, replace=False)
        return self.load_batch(batch_3D, batch_our, is_testing=True)

    def load_batch(self, batch_3D, batch_our, is_testing=False):
        imgs_3D, imgs_our = list(), list()

        for name_3D, name_our in zip(batch_3D, batch_our):
            img_3D = self.imread(os.path.join(self.data_path, self.style_3D, name_3D))
            img_our = self.imread(os.path.join(self.data_path, self.style_our, name_our))

            if not is_testing and np.random.random() > 0.5:
                img_3D = np.fliplr(img_3D)
                img_our = np.fliplr(img_our)

            imgs_3D.append(img_3D)
            imgs_our.append(img_our)

        imgs_3D = np.float32(imgs_3D) / 127.5 - 1
        imgs_our = np.float32(imgs_our) / 127.5 - 1
        return imgs_3D, imgs_our

    def imread(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float)


if __name__ == '__main__':
    from configs.cf_style_gray_rb import config as conf

    dataloader = DataLoader(conf)

    imgs_3D, imgs_our = dataloader.load_random_images_pair(batch_size=1)


    for batch_i, (imgs_3D, imgs_our) in enumerate(dataloader.load_batch_train(conf.batch_size)):
        print(batch_i, np.shape(imgs_3D), np.shape(imgs_our))

