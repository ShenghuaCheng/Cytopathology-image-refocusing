import os
import numpy as np
import random
import cv2
from multiprocessing import dummy, cpu_count

class DataLoader:
    def __init__(self, conf):
        self.source_style = conf.source_style
        self.target_style = conf.target_style
        self.data_path = conf.data_path
        self.txt_path = conf.txt_path
        self.num_per_epoch = conf.nums_per_epoch
        self.img_res = conf.img_res
        self.data_ratio = conf.data_ratio

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

        self.list_source_train = self.__get_lines_from_txt(self.source_style, '_train')
        self.list_target_train = self.__get_lines_from_txt(self.target_style, '_train')

        self.list_source_val = self.__get_lines_from_txt(self.source_style, '_val')
        self.list_target_val = self.__get_lines_from_txt(self.target_style, '_val')

        random.shuffle(self.list_source_val)
        random.shuffle(self.list_target_val)

        random.shuffle(self.list_source_train)
        random.shuffle(self.list_target_train)
        self.list_source_train = self.list_source_train[: self.num_per_epoch]
        self.list_target_train = self.list_target_train[: self.num_per_epoch]

        self.list_target_all = self.__get_lines_from_txt(self.target_style, '')
        random.shuffle(self.list_target_all)

    def load_data(self, domain, batch_size=1, is_testing=False):
        # pre = None
        if domain is self.source_style:
            pre = os.path.join(self.data_path, self.source_style)
            names = self.list_source_train if is_testing else self.list_source_val
        else:
            pre = os.path.join(self.data_path, self.target_style)
            names = self.list_target_train if is_testing else self.list_target_val

        names = np.random.choice(names, size=batch_size)

        imgs = list()
        for name in names:
            img = self.imread(os.path.join(pre, name))
            img = cv2.resize(img, self.img_res, interpolation=cv2.INTER_CUBIC)
            if not is_testing and np.random.random() > 0.5:
                img = np.fliplr(img)
            imgs.append(img)

        imgs = np.float32(imgs) / 255.

        return imgs

    def load_batch_train(self, batch_size=1):
        path_source = self.list_source_train
        path_target = self.list_target_train
        self.n_batches = int(min(len(path_source), len(path_target)) / batch_size)
        total_samples = self.n_batches * batch_size
        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_source = np.random.choice(path_source, total_samples, replace=False)
        path_target = np.random.choice(path_target, total_samples, replace=False)
        return self.load_batch(path_source, path_target, batch_size)

    
    def load_batch_val(self, batch_size=1):
        path_source = self.list_source_val
        path_target = self.list_target_val
        return self.load_batch(path_source, path_target, batch_size, is_testing=True)

    def load_batch(self, path_source, path_target, batch_size=1, is_testing=False):

        for i in range(self.n_batches - 1):
            s_batch_size = int(self.data_ratio[0] / sum(self.data_ratio) * batch_size)
            t_batch_size = int(self.data_ratio[1] / sum(self.data_ratio) * batch_size)
            batch_source = path_source[i * s_batch_size: (i + 1) * s_batch_size]
            batch_target = path_target[i * t_batch_size: (i + 1) * t_batch_size]
            batch_label = self.list_target_all[i * batch_size: (i + 1) * batch_size]

            imgs_source, imgs_target, imgs_label = list(), list(), list()

            def img_handle(imgs, style, p, testing):
                img = self.imread(os.path.join(self.data_path, style, p))
                img = cv2.resize(img, self.img_res, interpolation=cv2.INTER_CUBIC)
                if not testing and np.random.random() > 0.5:
                    img = np.fliplr(img)
                imgs.append(img)

            pool = dummy.Pool(cpu_count())

            for p_source in batch_source:
                pool.apply_async(img_handle, args=(imgs_source, self.source_style, p_source, is_testing))

            for p_target in batch_target:
                pool.apply_async(img_handle, args=(imgs_target, self.target_style, p_target, is_testing))

            for p_label in batch_label:
                pool.apply_async(img_handle, args=(imgs_label, self.target_style, p_label, False))

            pool.close()
            pool.join()

            imgs_source = imgs_source + imgs_target
            random.shuffle(imgs_source)
            imgs_source = np.float32(imgs_source) / 255.
            imgs_label = np.float32(imgs_label) / 255.

            yield imgs_source, imgs_label

    def load_img(self, path):
        img = self.imread(path)
        img = cv2.resize(img, self.img_res, interpolation=cv2.INTER_CUBIC)
        img = img / 255.0
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float)
