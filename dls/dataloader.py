import cv2
import numpy as np
import random
import os

import platform
sys = platform.system()

class DataLoaderStyle():
    def __init__(self, style_3D, style_our, txt_path, data_path, num_per_epoch, img_res=(256, 256)):
        self.style_3D = style_3D
        self.style_our = style_our
        self.data_path = data_path
        self.txt_path = txt_path
        self.num_per_epoch = num_per_epoch
        self.img_res = img_res

    #     get list
        self.__get_list__()

    def __get_lines_from_txt(self, style, mode='train'):
        tmp_list = list()
        with open('%s/%s_%s.txt' % (self.txt_path, style, mode), 'r') as tmp:
            for line in tmp:
                line = line.strip()
                tmp_list.append(line)
        return tmp_list

    def __get_list__(self):

        self.list_3D_train = self.__get_lines_from_txt(self.style_3D, 'train')
        self.list_our_train = self.__get_lines_from_txt(self.style_our, 'train')

        self.list_3D_val = self.__get_lines_from_txt(self.style_3D, 'val')
        self.list_our_val = self.__get_lines_from_txt(self.style_our, 'val')

        random.shuffle(self.list_3D_train)
        random.shuffle(self.list_our_train)
        self.list_3D_train = self.list_3D_train[: self.num_per_epoch]
        self.list_our_train = self.list_our_train[: self.num_per_epoch]

    def load_data(self, domain, batch_size=1, is_testing=False):
        # pre = None
        if domain is 'style_3D':
            pre = os.path.join(self.data_path, self.style_3D)
            names = self.list_3D_val if is_testing else self.list_3D_train
        else:
            pre = os.path.join(self.data_path, self.style_our)
            names = self.list_our_val if is_testing else self.list_our_train

        names = np.random.choice(names, size=batch_size)

        imgs = []
        for name in names:
            img = self.imread(os.path.join(pre, name))
            if not is_testing:
                img = cv2.resize(img, self.img_res, interpolation=cv2.INTER_CUBIC)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = cv2.resize(img, self.img_res, interpolation=cv2.INTER_CUBIC)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        path_3D = self.list_3D_train if not is_testing else self.list_3D_val
        path_our = self.list_our_train if not is_testing else self.list_our_val

        self.n_batches = int(min(len(path_3D), len(path_our)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_3D = np.random.choice(path_3D, total_samples, replace=False)
        path_our = np.random.choice(path_our, total_samples, replace=False)

        for i in range(self.n_batches - 1):
            batch_3D = path_3D[i * batch_size: (i + 1) * batch_size]
            batch_our = path_our[i * batch_size: (i + 1) * batch_size]
            imgs_3D, imgs_our = [], []
            for p_3D, p_our in zip(batch_3D, batch_our):
                img_3D = self.imread(os.path.join(self.data_path, self.style_3D, p_3D))
                img_our = self.imread(os.path.join(self.data_path, self.style_our, p_our))

                img_3D = cv2.resize(img_3D, self.img_res, interpolation=cv2.INTER_CUBIC)
                img_our = cv2.resize(img_our, self.img_res, interpolation=cv2.INTER_CUBIC)

                if not is_testing and np.random.random() > 0.5:
                        img_3D = np.fliplr(img_3D)
                        img_our = np.fliplr(img_our)

                imgs_3D.append(img_3D)
                imgs_our.append(img_our)

            imgs_3D = np.array(imgs_3D) / 127.5 - 1.
            imgs_our = np.array(imgs_our) / 127.5 - 1.

            yield imgs_3D, imgs_our

    def load_img(self, path):
        img = self.imread(path)
        img = cv2.resize(img, self.img_res, interpolation=cv2.INTER_CUBIC)
        img = img / 127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float)


class DataLoaderClear:
    def __init__(self, style_A, style_B, txt_path, num_per_epoch, img_res=(256, 256)):
        self.style_A = style_A
        self.style_B = style_B
        self.txt_path = txt_path
        self.num_per_epoch = num_per_epoch
        self.img_res = img_res

        self.pathA = None
        self.pathB = None

        #     get list
        self.__get_list__()

    def __get_lines_from_txt(self, style, mode='train'):
        tmp_list = list()
        with open('%s/%s_%s.txt' % (self.txt_path, style, mode), 'r') as tmp:
            for line in tmp:
                line = line.strip()
                tmp_list.append(line)
        return tmp_list

    def __get_list__(self):

        self.list_A_train = self.__get_lines_from_txt(self.style_A, 'train')
        self.list_B_train = self.__get_lines_from_txt(self.style_B, 'train')

        self.list_A_val = self.__get_lines_from_txt(self.style_A, 'test')
        self.list_B_val = self.__get_lines_from_txt(self.style_B, 'test')

        random.shuffle(self.list_A_train)
        random.shuffle(self.list_B_train)
        self.list_A_train = self.list_A_train[: self.num_per_epoch]
        self.list_B_train = self.list_B_train[: self.num_per_epoch]

    def load_data(self, domain, batch_size=1, is_testing=False):
        if domain is 'style_A':
            path = self.list_A_train if is_testing else self.list_A_val
            lpath = self.pathB
        else:
            path = self.list_B_train if is_testing else self.list_B_val
            lpath = self.pathA

        batch_images = np.random.choice(path, size=batch_size)
        # batch_images = path[ : batch_size]

        imgs = []
        imgs_label = []
        for img_path in batch_images:
            name = img_path[img_path.rfind('/') + 1:]
            img = self.imread(img_path)
            img_label = self.imread(lpath + name)
            if not is_testing:
                img = cv2.resize(img, self.img_res, interpolation=cv2.INTER_CUBIC)
                img_label = cv2.resize(img_label, self.img_res, interpolation=cv2.INTER_CUBIC)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
                    img_label = np.fliplr(img_label)
            else:
                img = cv2.resize(img, self.img_res, interpolation=cv2.INTER_CUBIC)
                img_label = cv2.resize(img_label, self.img_res, interpolation=cv2.INTER_CUBIC)
            imgs.append(img)
            imgs_label.append(img_label)

        imgs = np.array(imgs) / 127.5 - 1.
        imgs_label = np.array(imgs_label)/ 127.5 - 1.

        return imgs , imgs_label

    def load_batch(self, batch_size=1, is_testing=False):
        path_A = self.list_A_train if not is_testing else self.list_A_test
        path_B = self.list_B_train if not is_testing else self.list_B_test

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches - 1):
            batch_A = path_A[i * batch_size:(i + 1) * batch_size]
            batch_B = path_B[i * batch_size:(i + 1) * batch_size]
            imgs_A, imgs_B = [], []
            imgs_A_label, imgs_B_label = [], []
            for p_A, p_B in zip(batch_A, batch_B):
                name_A = p_A[p_A.rfind('/') + 1:]
                name_B = p_B[p_B.rfind('/') + 1:]
                img_A = self.imread(os.path.join(self.data_path, self.style_A, p_A))
                img_B = self.imread(os.path.join(self.data_path, self.style_A, p_B))
                if self.pathA is None:
                    self.pathA = p_A[: p_A.rfind('/') + 1]
                    self.pathB = p_B[: p_B.rfind('/') + 1]
                img_A_label = self.imread(self.pathB + name_A)
                img_B_label = self.imread(self.pathA + name_B)

                img_A = cv2.resize(img_A, self.img_res, interpolation=cv2.INTER_CUBIC)
                img_B = cv2.resize(img_B, self.img_res, interpolation=cv2.INTER_CUBIC)
                img_A_label = cv2.resize(img_A_label, self.img_res, interpolation=cv2.INTER_CUBIC)
                img_B_label = cv2.resize(img_B_label, self.img_res, interpolation=cv2.INTER_CUBIC)

                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)
                    img_A_label = np.fliplr(img_A_label)
                    img_B_label = np.fliplr(img_B_label)

                imgs_A.append(img_A)
                imgs_B.append(img_B)
                imgs_A_label.append(img_A_label)
                imgs_B_label.append(img_B_label)

            imgs_A = np.array(imgs_A) / 127.5 - 1.
            imgs_B = np.array(imgs_B) / 127.5 - 1.
            imgs_A_label = np.array(imgs_A_label) / 127.5 - 1.
            imgs_B_label = np.array(imgs_B_label) / 127.5 - 1.

            yield imgs_A, imgs_B, imgs_A_label , imgs_B_label

    def load_img(self, path):
        img = self.imread(path)
        img = cv2.resize(img, self.img_res, interpolation=cv2.INTER_CUBIC)
        img = img / 127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        if sys == "Windows":
            path = path.replace('/mnt/disk_8t/gxb/', 'D:/')

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float)

class DataLoaderClearDu():
    def __init__(self, data_path , blur, clear, txt_path, num_per_epoch, img_res=(256, 256)):
        self.data_path = data_path
        self.blur = blur
        self.clear = clear
        self.txt_path = txt_path
        self.num_per_epoch = num_per_epoch
        self.img_res = img_res

        #     get list
        self.__get_list__()

    def __get_lines_from_txt(self, style, mode='train'):
        tmp_list = list()
        with open('%s/%s_%s.txt' % (self.txt_path, style, mode), 'r') as tmp:
            for line in tmp:
                line = line.strip()
                tmp_list.append(line)
        return tmp_list

    def __get_list__(self):

        self.list_bl_train = self.__get_lines_from_txt(self.blur, 'train')
        self.list_bl_test = self.__get_lines_from_txt(self.blur, 'test')

        random.shuffle(self.list_bl_train)

        self.list_bl_train = self.list_bl_train[: self.num_per_epoch]

    def load_data(self, batch_size=1, is_testing=False):
        path = self.list_bl_train if is_testing else self.list_bl_test

        batch_images = np.random.choice(path, size=batch_size)

        bl_imgs , cl_imgs = list() , list()
        for img_path in batch_images:
            name = img_path[img_path.rfind('/') + 1:]
            bl_img = self.imread('%s/%s/%s' % (self.data_path, self.blur , name))
            cl_img = self.imread('%s/%s/%s' % (self.data_path, self.clear, name))

            bl_img = cv2.resize(bl_img, self.img_res, interpolation=cv2.INTER_CUBIC)
            cl_img = cv2.resize(cl_img, self.img_res, interpolation=cv2.INTER_CUBIC)
            if not is_testing and np.random.random() > 0.5:
                bl_img = np.fliplr(bl_img)
                cl_img = np.fliplr(cl_img)
            bl_imgs.append(bl_img)
            cl_imgs.append(cl_img)

        bl_imgs = np.array(bl_imgs) / 127.5 - 1.
        cl_imgs = np.array(cl_imgs) / 127.5 - 1.

        return bl_imgs , cl_imgs

    def load_batch(self, batch_size=1, is_testing=False):
        paths = self.list_bl_train if not is_testing else self.list_bl_test

        self.n_batches = int(len(paths) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        paths = np.random.choice(paths, total_samples, replace=False)

        for i in range(self.n_batches - 1):
            batch_path = paths[i * batch_size:(i + 1) * batch_size]
            bl_imgs, cl_imgs = [], []
            for p in batch_path:
                name = p[p.rfind('/') + 1:]
                bl_img = self.imread('%s/%s/%s' % (self.data_path, self.blur, name))
                cl_img = self.imread('%s/%s/%s' % (self.data_path, self.clear, name))
                bl_img = cv2.resize(bl_img, self.img_res, interpolation=cv2.INTER_CUBIC)
                cl_img = cv2.resize(cl_img, self.img_res, interpolation=cv2.INTER_CUBIC)
                if not is_testing and np.random.random() > 0.5:
                    bl_img = np.fliplr(bl_img)
                    cl_img = np.fliplr(cl_img)

                bl_imgs.append(bl_img)
                cl_imgs.append(cl_img)

            bl_imgs = np.array(bl_imgs) / 127.5 - 1.
            cl_imgs = np.array(cl_imgs) / 127.5 - 1.

            yield bl_imgs , cl_imgs

    def imread(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float)

class DataLoaderClearDuMix():
    def __init__(self, data_path, blur, clear, txt_path, num_per_epoch, img_res=(256, 256)):
        self.data_path = data_path
        self.blur = blur
        self.clear = clear
        self.txt_path = txt_path
        self.num_per_epoch = num_per_epoch
        self.img_res = img_res

        #     get list
        self.__get_list__()

    def __get_lines_from_txt(self, style, mode='train'):
        tmp_list = list()
        with open('%s/%s_%s.txt' % (self.txt_path, style, mode), 'r') as tmp:
            for line in tmp:
                line = line.strip()
                tmp_list.append(line)
        return tmp_list

    def __get_list__(self):

        self.list_bl1_train = self.__get_lines_from_txt(self.blur[0], 'train')
        self.list_bl1_test = self.__get_lines_from_txt(self.blur[0], 'test')

        self.list_bl2_train = self.__get_lines_from_txt(self.blur[1], 'train')
        self.list_bl2_test = self.__get_lines_from_txt(self.blur[1] , 'test')

        random.shuffle(self.list_bl1_train)
        random.shuffle(self.list_bl2_train)

        self.list_bl1_train = self.list_bl1_train[: self.num_per_epoch]
        self.list_bl2_train = self.list_bl2_train[: self.num_per_epoch]

    def load_data(self, batch_size=1, is_testing=False):
        path1 = self.list_bl1_train if is_testing else self.list_bl1_test
        path2 = self.list_bl2_train if is_testing else self.list_bl2_test

        batch_images1 = np.random.choice(path1, size=batch_size)
        batch_images2 = np.random.choice(path2, size=batch_size)

        bl1_imgs , bl2_imgs, cl_imgs = list() , list() , list()
        for k in range(len(batch_images1)):
            name1 = batch_images1[k][batch_images1[k].rfind('/') + 1:]
            name2 = batch_images2[k][batch_images2[k].rfind('/') + 1:]
            bl1_img = self.imread('%s/%s/%s' % (self.data_path, self.blur[0] , name1))
            bl2_img = self.imread('%s/%s/%s' % (self.data_path, self.blur[1] , name2))
            cl_img = self.imread('%s/%s/%s' % (self.data_path, self.clear, name1))

            bl1_img = cv2.resize(bl1_img, self.img_res, interpolation=cv2.INTER_CUBIC)
            bl2_img = cv2.resize(bl2_img, self.img_res, interpolation=cv2.INTER_CUBIC)
            cl_img = cv2.resize(cl_img, self.img_res, interpolation=cv2.INTER_CUBIC)
            if not is_testing and np.random.random() > 0.5:
                bl1_img = np.fliplr(bl1_img)
                bl2_img = np.fliplr(bl2_img)
                cl_img = np.fliplr(cl_img)
            bl1_imgs.append(bl1_img)
            bl2_imgs.append(bl2_img)
            cl_imgs.append(cl_img)

        bl1_imgs = np.array(bl1_imgs) / 127.5 - 1.
        bl2_imgs = np.array(bl2_imgs) / 127.5 - 1.
        cl_imgs = np.array(cl_imgs) / 127.5 - 1.

        return bl1_imgs ,bl2_imgs , cl_imgs

    def load_batch(self, batch_size=1, is_testing=False):
        path1s = self.list_bl1_train if not is_testing else self.list_bl1_test
        path2s = self.list_bl2_train if not is_testing else self.list_bl2_test

        self.n_batches = int(min(len(path1s) , len(path2s)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path1s = np.random.choice(path1s, total_samples, replace=False)
        path2s = np.random.choice(path2s, total_samples, replace=False)

        for i in range(self.n_batches - 1):
            batch_path1 = path1s[i * batch_size:(i + 1) * batch_size]
            batch_path2 = path2s[i * batch_size:(i + 1) * batch_size]
            bl1_imgs, bl2_imgs, cl_imgs = [], [], []
            for k in range(len(batch_path1)):
                name1 = batch_path1[k][batch_path1[k].rfind('/') + 1:]
                name2 = batch_path2[k][batch_path2[k].rfind('/') + 1:]
                bl1_img = self.imread('%s/%s/%s' % (self.data_path, self.blur[0], name1))
                bl2_img = self.imread('%s/%s/%s' % (self.data_path, self.blur[1], name2))
                cl_img = self.imread('%s/%s/%s' % (self.data_path, self.clear, name1))
                bl1_img = cv2.resize(bl1_img, self.img_res, interpolation=cv2.INTER_CUBIC)
                bl2_img = cv2.resize(bl2_img, self.img_res, interpolation=cv2.INTER_CUBIC)
                cl_img = cv2.resize(cl_img, self.img_res, interpolation=cv2.INTER_CUBIC)
                if not is_testing and np.random.random() > 0.5:
                    bl1_img = np.fliplr(bl1_img)
                    bl2_img = np.fliplr(bl2_img)
                    cl_img = np.fliplr(cl_img)

                bl1_imgs.append(bl1_img)
                bl2_imgs.append(bl2_img)
                cl_imgs.append(cl_img)

            bl1_imgs = np.array(bl1_imgs) / 127.5 - 1.
            bl2_imgs = np.array(bl2_imgs) / 127.5 - 1.
            cl_imgs = np.array(cl_imgs) / 127.5 - 1.

            yield bl1_imgs , bl2_imgs, cl_imgs

    def imread(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float)


class DataLoaderClearMix():
    def __init__(self, style_A, style_B, txt_path, num_per_epoch, img_res=(256, 256) , data_ratio = [0.5 , 0.5]):
        self.style_A = style_A
        self.style_B = style_B
        self.txt_path = txt_path
        self.num_per_epoch = num_per_epoch
        self.img_res = img_res
        self.data_ratio = data_ratio

        self.pathA = None
        self.pathB = None

        #     get list
        self.__get_list__()

    def __get_lines_from_txt(self, style, mode='train'):
        tmp_list = list()
        with open('%s/%s_%s.txt' % (self.txt_path, style, mode), 'r') as tmp:
            for line in tmp:
                line = line.strip()
                tmp_list.append(line)
        return tmp_list

    def __get_list__(self):

        # A 风格的数据由3D和经过our转换成3D的数据构成
        self.list_A1_train = self.__get_lines_from_txt(self.style_A[0], 'train')
        self.list_A2_train = self.__get_lines_from_txt(self.style_A[1], 'train')

        self.list_B_train = self.__get_lines_from_txt(self.style_B, 'train')

        self.list_A1_test = self.__get_lines_from_txt(self.style_A[0], 'test')
        self.list_A2_test = self.__get_lines_from_txt(self.style_A[1], 'test')

        self.list_B_test = self.__get_lines_from_txt(self.style_B, 'test')

        random.shuffle(self.list_A1_train)
        random.shuffle(self.list_A2_train)
        random.shuffle(self.list_B_train)

        self.list_A1_train = self.list_A1_train[ : int(self.num_per_epoch * self.data_ratio[0])]
        self.list_A2_train = self.list_A2_train[ : int(self.num_per_epoch * self.data_ratio[1])]

        self.list_B_train = self.list_B_train[: self.num_per_epoch]

    def load_data(self, domain, batch_size=1, is_testing=False):
        if domain is 'style_A':
            if random.randint(0 , 1):
                path = self.list_A1_train if is_testing else self.list_A1_test
            else:
                path = self.list_A2_train if is_testing else self.list_A2_test
            lpath = self.pathB
        else:
            path = self.list_B_train if is_testing else self.list_B_test
            lpath = self.pathA

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        imgs_label = []
        for img_path in batch_images:
            name = img_path[img_path.rfind('/') + 1:]
            # print(img_path , lpath + name)
            img = self.imread(img_path)
            img_label = self.imread(lpath + name)
            if not is_testing:
                img = cv2.resize(img, self.img_res, interpolation=cv2.INTER_CUBIC)
                if img_label is not None:
                    img_label = cv2.resize(img_label, self.img_res, interpolation=cv2.INTER_CUBIC)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
                    if img_label is not None:
                        img_label = np.fliplr(img_label)
            else:
                img = cv2.resize(img, self.img_res, interpolation=cv2.INTER_CUBIC)
                if img_label is not None:
                    img_label = cv2.resize(img_label, self.img_res, interpolation=cv2.INTER_CUBIC)
            imgs.append(img)
            imgs_label.append(img_label)

        imgs = np.array(imgs) / 127.5 - 1.
        if img_label is not None:
            imgs_label = np.array(imgs_label)/ 127.5 - 1.

        return imgs , imgs_label

    def load_batch(self, batch_size=1, is_testing=False):
        path_A1 = self.list_A1_train if not is_testing else self.list_A1_test
        path_A2 = self.list_A2_train if not is_testing else self.list_A2_test

        path_B = self.list_B_train if not is_testing else self.list_B_test

        self.n_batches = int(self.num_per_epoch / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A1 = np.random.choice(path_A1, int(total_samples * self.data_ratio[0]), replace=False)
        path_A2 = np.random.choice(path_A2, int(total_samples * self.data_ratio[1]), replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        flag = True
        ind1 = 0
        ind2 = 0
        for i in range(self.n_batches - 1):
            if flag:
                batch_A = path_A1[ind1 * batch_size:(ind1 + 1) * batch_size]
            else:
                batch_A = path_A2[ind2 * batch_size:(ind2 + 1) * batch_size]
            batch_B = path_B[i * batch_size:(i + 1) * batch_size]

            imgs_A, imgs_B = [], []
            imgs_A_label, imgs_B_label = [], []
            for p_A, p_B in zip(batch_A, batch_B):
                name_A = p_A[p_A.rfind('/') + 1:]
                name_B = p_B[p_B.rfind('/') + 1:]
                img_A = self.imread(p_A)
                img_B = self.imread(p_B)
                if self.pathA is None:
                    self.pathA = p_A[: p_A.rfind('/') + 1]
                    self.pathB = p_B[: p_B.rfind('/') + 1]
                img_A_label = self.imread(self.pathB + name_A)
                img_B_label = self.imread(p_B[ : p_B.find('_label/')] + '/' + name_B)

                img_A = cv2.resize(img_A, self.img_res, interpolation=cv2.INTER_CUBIC)
                img_B = cv2.resize(img_B, self.img_res, interpolation=cv2.INTER_CUBIC)
                if img_A_label is not None:
                    img_A_label = cv2.resize(img_A_label, self.img_res, interpolation=cv2.INTER_CUBIC)
                img_B_label = cv2.resize(img_B_label, self.img_res, interpolation=cv2.INTER_CUBIC)

                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)
                    if img_A_label is not None:
                        img_A_label = np.fliplr(img_A_label)
                    img_B_label = np.fliplr(img_B_label)

                imgs_A.append(img_A)
                imgs_B.append(img_B)
                imgs_A_label.append(img_A_label)
                imgs_B_label.append(img_B_label)

            imgs_A = np.array(imgs_A) / 127.5 - 1.
            imgs_B = np.array(imgs_B) / 127.5 - 1.
            if imgs_A_label[0] is not None:
                imgs_A_label = np.array(imgs_A_label) / 127.5 - 1.
            imgs_B_label = np.array(imgs_B_label) / 127.5 - 1.

            if flag:
                ind1 += 1
            else:
                ind2 += 1
            flag = not flag
            yield imgs_A, imgs_B, imgs_A_label , imgs_B_label

    def load_img(self, path):
        img = self.imread(path)
        img = cv2.resize(img, self.img_res, interpolation=cv2.INTER_CUBIC)
        img = img / 127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        if sys == "Windows":
            path = path.replace('/mnt/disk_8t/gxb/' , 'W:/')

        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float)


if __name__ == '__main__':
    from train_config import clear_stage1_cf as conf
    dataloader = DataLoaderClearMix(conf.style_A, conf.style_B, conf.txt_path, conf.nums_per_epoch, conf.img_shape[: 2])
    for batch_i, (imgs_A, imgs_B, imgs_Al, imgs_Bl) in enumerate(dataloader.load_batch(conf.batch_size)):


        imgs_A, imgs_Al = dataloader.load_data(domain='style_A', batch_size=conf.batch_size, is_testing=True)
        imgs_B, imgs_Bl = dataloader.load_data(domain='style_B', batch_size=conf.batch_size, is_testing=True)