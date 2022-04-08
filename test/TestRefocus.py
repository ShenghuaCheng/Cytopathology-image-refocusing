'''
test img clear model
'''
from test.cfs_test.cf_t_clear import config as conf
import os
from mmodels.mb import g_nets_dict
from libs.datautils import pre_op, suf_op
from libs.metrics import cal_metrics
from train.trainer import trainer

import datetime
import cv2
from tqdm import tqdm
import numpy as np


class Refocus:

    def __init__(self, conf):

        self.img_save_flag = conf.img_save_flag
        self.data_path = conf.img_path
        self.save_path = conf.save_path
        self.item = conf.item
        self.detail = conf.detail
        self.scale_num = conf.scale_num
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # self.__init_model__(conf)
        self.__read_name_list__(conf.txt_path, conf.test_nums)
        # self.__load_imgs__(conf.img_path, conf.item, conf.detail, conf.scale_num)


    def __init_model__(self, conf):
        self.g_BS = g_nets_dict[conf.g_net](conf)


    def __load_model__(self, weights_path):
        self.g_BS.load_weights(weights_path)

    def __read_name_list__(self, txt_path, test_num=100):
        '''
        :param txt_path: the txt path to read
        :return: nothing
        '''
        self.name_list = list()
        with open(txt_path, 'r') as test_img_txt:
            for line in test_img_txt:
                line = line.strip()
                self.name_list.append(line)

        self.name_list = self.name_list[:test_num]

    def __reshape__(self, img):
        return np.reshape(img, (1,) + img.shape)

    def __load_imgs__(self, img_path, item, detail):
        '''
        :param img_path:
        :param label_path:
        :param detail:
        :return:
        '''
        self.imgs = list()
        self.nor_imgs = list()
        self.lab_imgs = list()
        self.img_kers = list()
        self.img_recs = list()

        print('load img ....')
        for name in tqdm(self.name_list):

            img, nor_img, lab_img = self.__load_item__(img_path, item, name)
            self.imgs.append(img)
            self.lab_imgs.append(lab_img)
            self.nor_imgs.append(nor_img)

            if detail:
                img_ker, img_rec = self.__load_detail__(img, img_path, item, name)
                self.img_kers.append(img_ker)
                self.img_recs.append(img_rec)

    def __load_item__(self, img_path, item, name):

        img = cv2.imread(os.path.join(img_path, item, name))
        lab_img = cv2.imread(os.path.join(img_path, '3D_label', name))

        # normalization the input img
        nor_img = pre_op(img)
        nor_img = self.__reshape__(nor_img)

        return img, nor_img, lab_img

    def __load_detail__(self, img, img_path, item, name):
        img_ker, img_rec = trainer.block1(img, img_path, item, name)
        img_ker = img_ker[: self.scale_num]
        img_rec = img_rec[: self.scale_num]
        img_ker = img_ker / 127.5 - 1

        img_ker = self.__reshape__(img_ker)
        img_rec = self.__reshape__(img_rec)
        return img_ker, img_rec


    def test_speed(self):
        '''
        test inference speed of refocus model
        :return: no return
        '''
        self.g_BS.summary()
        start_time = datetime.datetime.now()

        # prepare model input
        inputs = []
        for nor_img, img_ker, img_rec in zip(self.nor_imgs, self.img_kers, self.img_recs):
            inputs.append([nor_img, img_ker, img_rec])

        for input in tqdm(inputs):
            result = self.g_BS.predict_on_batch(input)
        # print(len(results))
        elapsed_time = datetime.datetime.now() - start_time
        print('%s, ts: %s' % (weight_path, elapsed_time))


    def test(self):
        '''
        :return:  get SSIM and PSNR of refocus image
        '''


        start_time = datetime.datetime.now()

        metrics_dict = {}
        for me in conf.metrics:
            metrics_dict[me] = []

        for ind in tqdm(range(0, len(self.name_list))):
            img, nor_img, lab_img = self.__load_item__(self.data_path, self.item, self.name_list[ind])
            if conf.detail:
                img_ker, img_rec = self.__load_detail__(img, self.data_path, self.item, self.name_list[ind])
                gen_img = self.g_BS.predict([nor_img, img_ker, img_rec])[0]
            else:
                gen_img = self.g_BS.predict(nor_img)[0]
            gen_img = suf_op(gen_img)

            if self.img_save_flag:
                # nor_img = suf_op(nor_img[0, :, :, :])
                # combine = cv2.hconcat((img, gen_img, label))
                cv2.imwrite(os.path.join(self.save_path, self.name_list[ind]), gen_img)

            re_dict = cal_metrics(gen_img, lab_img, choose_metrics=conf.metrics)
            for me in conf.metrics:
                metrics_dict[me].append(re_dict[me])

        elapsed_time = datetime.datetime.now() - start_time
        for me in conf.metrics:
            print('%s: %.4f, ' % (me, np.mean(metrics_dict[me])), end='')
        print('%s, ts: %s' % (weight_path, elapsed_time))

    def test_output(self):
        '''
        :return:  get output of split img and kernel
        '''

        save_path = r'D:\paper_stage2\log\clear_stage1_nocircle(unet&multi_scale&3kernel)_v1\tmp'

        metrics_dict = {}
        for me in conf.metrics:
            metrics_dict[me] = []

        for ind in tqdm(range(0, len(self.imgs))):
            img, nor_img, lab_img = self.imgs[ind], self.nor_imgs[ind], self.lab_imgs[ind]

            g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            g_lab_img = cv2.cvtColor(lab_img, cv2.COLOR_BGR2GRAY)
            g_dis_img = np.abs(g_img - g_lab_img)

            g_dis_img = cv2.cvtColor(g_dis_img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(save_path, '%s_in.jpg' % self.name_list[ind]), img)
            cv2.imwrite(os.path.join(save_path, '%s_la.jpg' % self.name_list[ind]), lab_img)
            cv2.imwrite(os.path.join(save_path, '%s_dis.jpg' % self.name_list[ind]), g_dis_img)

            if conf.detail:
                img_ker, img_rect = self.img_kers[ind], self.img_recs[ind]

                for i in range(conf.scale_num):
                    k = suf_op(img_ker[0, i, :, :, :])
                    k = cv2.cvtColor(k, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite(os.path.join(save_path, '%d_%d_kin.jpg' % (ind, i)), k)

                outputs = []
                gen_imgs = self.g_BS.predict([nor_img, img_ker, img_rect])
                for gen_img in gen_imgs:
                    outputs.append(suf_op(gen_img[0, :, :, :]))

                cv2.imwrite(os.path.join(save_path, '%s_512.jpg' % (self.name_list[ind], )), cv2.cvtColor(outputs[1], cv2.COLOR_BGR2GRAY))

                cv2.imwrite(os.path.join(save_path, '%s_%d_kout.jpg' % (self.name_list[ind], 0)), outputs[2])
                cv2.imwrite(os.path.join(save_path, '%s_%d_kout.jpg' % (self.name_list[ind], 1)), outputs[3])
                cv2.imwrite(os.path.join(save_path, '%s_%d_kout.jpg' % (self.name_list[ind], 2)), outputs[4])


    def test_kernel(self,):
        '''
        :return:  get SSIM and PSNR of refocus kernel
        '''
        tmp_path = r'D:\paper_stage2\log\clear_stage1(unet&multi_scale)\tmp'

        metrics_dict = {}
        for me in conf.metrics:
            metrics_dict[me] = []

        label = '3D_label'
        for ind in tqdm(range(0, len(self.name_list))):
            gen_img = cv2.imread(os.path.join(self.save_path, self.name_list[ind]))
            lab_img = cv2.imread(os.path.join(self.data_path, label, self.name_list[ind]))

            # find kernel from images
            ker_imgs_lab, _ = trainer.find_kernels(lab_img, self.data_path, self.item, self.name_list[ind])
            ker_imgs_gen, _ = trainer.find_kernels(gen_img, self.data_path, self.item, self.name_list[ind])

            # calculate metrics of kernels
            k = 0
            for ker_img_lab, ker_img_gen in zip(ker_imgs_lab, ker_imgs_gen):
                gen_save_name = '%s_%d_%s.jpg' % (self.name_list[ind], k, conf.test_group)
                lab_save_name = '%s_%d_%s.jpg' % (self.name_list[ind], k, label)
                cv2.imwrite(os.path.join(tmp_path, gen_save_name), ker_img_gen)
                cv2.imwrite(os.path.join(tmp_path, lab_save_name), ker_img_lab)
                k += 1
                continue
                re_dict = cal_metrics(ker_img_gen, ker_img_lab, choose_metrics=conf.metrics)
                # print(re_dict)
                for me in conf.metrics:
                    metrics_dict[me].append(re_dict[me])

        for me in conf.metrics:
            print('%s: %.4f, ' % (me, np.mean(metrics_dict[me])), end='')



if __name__ == '__main__':

    # Specify gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpus

    rfn = Refocus(conf)
    rfn.__load_imgs__(conf.img_path, conf.item, conf.detail)
    rfn.__init_model__(conf)

    for model in conf.test_models:
        weight_path = os.path.join(conf.weig_path, model)
        print(weight_path)
        rfn.__load_model__(weight_path)
        rfn.test_speed()
        # rfn.test()

        # rfn.test_kernel()