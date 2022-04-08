import random
import numpy as np
import cv2
import os

from multiprocessing import dummy, cpu_count
from libs.datautils import get_binimg
from train.trainer import trainer

class DataLoaderClear:

    def __init__(self, conf):
        self.style_B = conf.style_B
        self.style_S = conf.style_S
        self.data_path = conf.data_path
        self.txt_path = conf.txt_path
        self.num_per_epoch = conf.nums_per_epoch
        self.img_res = conf.img_res

        self.att_flag = conf.att_loss

        # attention mask config
        if self.att_flag:
            self.att_path = conf.att_path
            self.col_thre = conf.col_thre
            self.vol_thre = conf.vol_thre
            
        self.detail = conf.detail
        self.scale_num = conf.scale_num

        self.pathB = None
        self.pathS = None

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

        self.list_B_train = self.__get_lines_from_txt(self.style_B, '_train')
        self.list_S_train = self.__get_lines_from_txt(self.style_S, '_train')
        print('read train txt(%s: %d, %s, %d)' % (self.style_B, len(self.list_B_train),
                                                  self.style_S, len(self.list_S_train)))
        self.list_B_val = self.__get_lines_from_txt(self.style_B, '_val')
        self.list_S_val = self.__get_lines_from_txt(self.style_S, '_val')
        print('read val txt(%s: %d, %s, %d)' % (self.style_B, len(self.list_B_val),
                                                self.style_S, len(self.list_S_val)))

        random.shuffle(self.list_B_train)
        random.shuffle(self.list_S_train)
        self.list_B_train = self.list_B_train[: self.num_per_epoch]
        self.list_S_train = self.list_S_train[: self.num_per_epoch]
        print('data train per epoch(%s: %d, %s, %d)' % (self.style_B, len(self.list_B_train),
                                                        self.style_S, len(self.list_S_train)))

        # random.shuffle(self.list_B_val)
        # random.shuffle(self.list_S_val)


    def load_batch_train(self, batch_size=1, is_testing=False):
        path_B = self.list_B_train
        path_S = self.list_S_train
        self.n_batches = int(min(len(path_B), len(path_S)) / batch_size)
        total_samples = self.n_batches * batch_size
        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_B = np.random.choice(path_B, total_samples, replace=False)
        path_S = np.random.choice(path_S, total_samples, replace=False)

        return self.load_batch(path_B, path_S, batch_size, is_testing)


    def load_batch_val(self, batch_size=1, is_testing=True, random=True):
        path_B = self.list_B_val
        path_S = self.list_S_val
        if random:
            path_B = np.random.choice(path_B, batch_size, replace=False)
            path_S = np.random.choice(path_S, batch_size, replace=False)
        else:
            path_B = path_B[: batch_size]
            path_S = path_S[: batch_size]

        list_dict = self.__generate_list_dict__()
        pool = dummy.Pool(cpu_count())
        for p_B, p_S in zip(path_B, path_S):
            pool.apply_async(self.__read_img_item__, args=(p_B, p_S, list_dict, is_testing))
        pool.close()
        pool.join()

        for k in list_dict.keys():
            list_dict[k] = np.float32(list_dict[k]) / 127.5 - 1

        return list_dict


    def __generate_list_dict__(self):
        list_dict = dict()
        list_dict['imgs_B'] = list()
        list_dict['imgs_S'] = list()
        list_dict['imgs_B_label'] = list()
        list_dict['imgs_S_label'] = list()
        if self.att_flag:
            list_dict['imgs_B_att'] = list()
            list_dict['imgs_S_att'] = list()
        if self.detail: # add detail branch
            list_dict['imgs_B_det'] = list()
            list_dict['imgs_B_rec'] = list()
            list_dict['imgs_S_det'] = list()
            list_dict['imgs_S_rec'] = list()
        return list_dict

    def __read_img_item__(self, p_B, p_S, list_dict, is_testing=False):
        '''
        :param p_B: blur img name
        :param p_S: sharp img name
        :param list_dict: img list dict
        :param is_testing:
        :return: no return
        '''
        img_B = self.imread(os.path.join(self.data_path, self.style_B, p_B))
        img_S = self.imread(os.path.join(self.data_path, self.style_S, p_S))

        img_B_label = self.imread(os.path.join(self.data_path, self.style_S, p_B))
        img_S_label = self.imread(os.path.join(self.data_path, self.style_B, p_S))

        if self.att_flag:
            img_B_att = self.imread(os.path.join(self.att_path, p_B[:p_B.find('.png')] + '_PC.png'))
            img_B_att = np.mean(img_B_att, axis=-1)
            img_B_att = img_B_att * get_binimg(img_B, self.col_thre, self.vol_thre)

            img_S_att = self.imread(os.path.join(self.att_path, p_S[:p_S.find('.png')] + '_PC.png'))
            img_S_att = np.mean(img_S_att, axis=-1)
            img_S_att = img_S_att * get_binimg(img_S, self.col_thre, self.vol_thre)

        if self.detail:
            img_B_ker, img_B_rect = trainer.block1(img_B, self.data_path, self.style_B, p_B)
            img_S_ker, img_S_rect = trainer.block1(img_S, self.data_path, self.style_B, p_S)
            list_dict['imgs_B_det'].append(img_B_ker[: self.scale_num])
            list_dict['imgs_B_rec'].append(img_B_rect[: self.scale_num])
            list_dict['imgs_S_det'].append(img_S_ker[: self.scale_num])
            list_dict['imgs_S_rec'].append(img_S_rect[: self.scale_num])

        # if not is_testing and np.random.random() > 0.5:
        if False:
            img_B = np.fliplr(img_B)
            img_S = np.fliplr(img_S)
            img_B_label = np.fliplr(img_B_label)
            img_S_label = np.fliplr(img_S_label)
            if self.att_flag:
                img_B_att = np.fliplr(img_B_att)
                img_S_att = np.fliplr(img_S_att)

        if self.att_flag:
            img_B_att = np.reshape(img_B_att, img_B_att.shape + (1, ))
            img_S_att = np.reshape(img_S_att, img_S_att.shape + (1,))

        list_dict['imgs_B'].append(img_B)
        list_dict['imgs_S'].append(img_S)
        list_dict['imgs_B_label'].append(img_B_label)
        list_dict['imgs_S_label'].append(img_S_label)

        if self.att_flag:
            list_dict['imgs_B_att'].append(img_B_att)
            list_dict['imgs_S_att'].append(img_S_att)


    def load_batch(self, path_B, path_S, batch_size, is_testing):

        for i in range(self.n_batches - 1):
            batch_B = path_B[i * batch_size: (i + 1) * batch_size]
            batch_S = path_S[i * batch_size: (i + 1) * batch_size]
            list_dict = self.__generate_list_dict__()
            pool = dummy.Pool(cpu_count())
            for p_B, p_S in zip(batch_B, batch_S):
                pool.apply_async(self.__read_img_item__, args=(p_B, p_S, list_dict, is_testing))
            pool.close()
            pool.join()

            for k in list_dict.keys():
                if k.find('rect') == -1:
                    list_dict[k] = np.float32(list_dict[k]) / 127.5 - 1.

            yield list_dict


    def imread(self, path):
        # if self.data_dict
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float)


