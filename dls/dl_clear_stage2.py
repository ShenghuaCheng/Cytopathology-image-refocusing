import os
import numpy as np
import random
import cv2
import multiprocessing
from multiprocessing import dummy, cpu_count
from libs.datautils import get_binimg
from train.trainer import trainer

class DataLoader:
    def __init__(self, conf):
        self.style_B = conf.style_B
        self.style_B_ratio = conf.style_B_ratio
        self.style_S = conf.style_S
        self.data_path = conf.data_path
        self.txt_path = conf.txt_path
        self.num_per_epoch = conf.nums_per_epoch
        self.img_res = conf.img_res

        self.att_flag = conf.att_loss
        if self.att_flag:
            self.att_path = conf.att_path
            self.col_thre = conf.col_thre
            self.vol_thre = conf.vol_thre

        self.detail = conf.detail
        self.scale_num = conf.scale_num

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

        self.list_B1_train = self.__get_lines_from_txt(self.style_B[0], '_train')
        self.list_B2_train = self.__get_lines_from_txt(self.style_B[1], '_train')
        self.list_S_train = self.__get_lines_from_txt(self.style_S, '_train')

        self.list_B1_val = self.__get_lines_from_txt(self.style_B[0], '_val')
        self.list_B2_val = self.__get_lines_from_txt(self.style_B[1], '_val')
        self.list_S_val = self.__get_lines_from_txt(self.style_S, '_val')

        random.shuffle(self.list_B1_val)
        random.shuffle(self.list_B2_val)
        random.shuffle(self.list_S_val)

        random.shuffle(self.list_B1_train)
        random.shuffle(self.list_B2_train)
        random.shuffle(self.list_S_train)

        self.list_B1_train = self.list_B1_train[: self.num_per_epoch]
        self.list_B2_train = self.list_B2_train[: self.num_per_epoch]
        self.list_S_train = self.list_S_train[: self.num_per_epoch]

    def load_batch_train(self, batch_size=1):
        path_B1 = self.list_B1_train
        path_B2 = self.list_B2_train
        path_S = self.list_S_train
        self.n_batches = int(self.num_per_epoch / batch_size)
        return self.load_batch1(path_B1, path_B2, path_S, batch_size)

    def choose_val_batch(self, random=True):
        if random:
            path_B1 = np.random.choice(self.list_B1_val, 1000, replace=False)
            path_B2 = np.random.choice(self.list_B2_val, 1000, replace=False)
            path_S = np.random.choice(self.list_S_val, 1000, replace=False)
        else:
            path_B1 = self.list_B1_val
            path_B2 = self.list_B2_val
            path_S = self.list_S_val
        return path_B1, path_B2, path_S

    def __read_att__(self, img, p):
        img_att = self.imread(os.path.join(self.att_path, p[:p.find('.png')] + '_PC.png'))
        img_att = np.mean(img_att, axis=-1)
        img_att = img_att * get_binimg(img, self.col_thre, self.vol_thre)
        return img_att

    def __generate_list_dict__(self):
        list_dict = dict()
        list_dict['imgs_B1'] = list()
        list_dict['imgs_B1_label'] = list()
        list_dict['imgs_B2'] = list()
        list_dict['imgs_S'] = list()
        list_dict['imgs_S_label'] = list()
        if self.detail:  # add detail branch
            list_dict['imgs_B1_det'] = list()
            list_dict['imgs_B1_rec'] = list()
            list_dict['imgs_B2_det'] = list()
            list_dict['imgs_B2_rec'] = list()
            list_dict['imgs_S_det'] = list()
            list_dict['imgs_S_rec'] = list()
        return list_dict


    def load_batch_val(self, batch_size=1, random=True):
        path_B1, path_B2, path_S = self.choose_val_batch(random)
        path_B1 = path_B1[: batch_size]
        path_B2 = path_B2[: batch_size]
        path_S = path_S[: batch_size]

        ld = self.__generate_list_dict__()
        for p_B1, p_B2, p_S in zip(path_B1, path_B2, path_S):
            img_B1 = self.imread(os.path.join(self.data_path, self.style_B[0], p_B1))
            img_B1_label = self.imread(os.path.join(self.data_path, self.style_S, p_B1))
            img_B2 = self.imread(os.path.join(self.data_path, self.style_B[1], p_B2))
            img_S = self.imread(os.path.join(self.data_path, self.style_S, p_S))
            img_S_label = self.imread(os.path.join(self.data_path, self.style_B[0], p_S))

            # add kernel detail for more information
            if self.detail:
                img_B1_ker, img_B1_rect = trainer.block1(img_B1, self.data_path, self.style_B[0], p_B1)
                img_B2_ker, img_B2_rect = trainer.block1(img_B2, self.data_path, self.style_B[1], p_B2)
                img_S_ker, img_S_rect = trainer.block1(img_S, self.data_path, self.style_B[0], p_S)

                #intercept scale_num
                ld['imgs_B1_det'].append(img_B1_ker[: self.scale_num])
                ld['imgs_B1_rec'].append(img_B1_rect[: self.scale_num])
                ld['imgs_B2_det'].append(img_B2_ker[: self.scale_num])
                ld['imgs_B2_rec'].append(img_B2_rect[: self.scale_num])
                ld['imgs_S_det'].append(img_S_ker[: self.scale_num])
                ld['imgs_S_rec'].append(img_S_rect[: self.scale_num])

            # add att_flag loss
            if self.att_flag:
                # 3DHistech att
                img_B1_att = self.__read_att__(img_B1, p_B1)
                img_B1 = np.concatenate([img_B1, img_B1_att.reshape(img_B1_att.shape + (1, ))], axis=-1)
                # 3DHistech label att
                img_S_att = self.__read_att__(img_S, p_S)
                img_S = np.concatenate([img_S, img_S_att.reshape(img_S_att.shape + (1, ))], axis=-1)

            ld['imgs_B1'].append(img_B1)
            ld['imgs_B1_label'].append(img_B1_label)
            ld['imgs_B2'].append(img_B2)
            ld['imgs_S'].append(img_S)
            ld['imgs_S_label'].append(img_S_label)


        for k in ld.keys():
            ld[k] = np.array(ld[k])
            if k.find('rec') == -1:
                ld[k] = ld[k] / 127.5 - 1.

        return ld

    def load_batch1(self, path_B1, path_B2, path_S, batch_size=1, is_testing=False):

        for i in range(self.n_batches - 1):
            b1_batch_size = int(self.style_B_ratio[0] * batch_size)
            b2_batch_size = int(self.style_B_ratio[1] * batch_size)
            b1_batch = path_B1[i * b1_batch_size: (i + 1) * b1_batch_size]
            b2_batch = path_B2[i * b2_batch_size: (i + 1) * b2_batch_size]

            s_batch = path_S[i * batch_size: (i + 1) * batch_size]

            ld = self.__generate_list_dict__()
            lock = multiprocessing.Lock()
            def img_handle(ks, style_list, p, testing=False):
                assert 0 < len(ks) < 5
                if len(ks) == 3:
                    img = self.imread(os.path.join(self.data_path, style_list[0], p))
                    # if not testing and np.random.random() > 0.5:
                    #     img = np.fliplr(img)

                    ld[ks[0]].append(img)
                    if self.detail:
                        img_ker, img_rect = trainer.block1(img, self.data_path, style_list[0], p)
                        # intercept scale_num
                        ld[ks[-2]].append(img_ker[: self.scale_num])
                        ld[ks[-1]].append(img_rect[: self.scale_num])

                if len(ks) == 4:
                    img1 = self.imread(os.path.join(self.data_path, style_list[0], p))
                    img2 = self.imread(os.path.join(self.data_path, style_list[1], p))
                    if self.att_flag:
                        img_att = self.__read_att__(img1, p)
                    # if not testing and np.random.random() > 0.5:
                    #     img1 = np.fliplr(img1)
                    #     img2 = np.fliplr(img2)
                    #     if self.att_flag:
                    #         img_att = np.fliplr(img_att)

                    if self.detail:
                        img_ker, img_rect = trainer.block1(img1, self.data_path, '3D_to_3D', p)
                        # intercept scale_num
                        ld[ks[-2]].append(img_ker[: self.scale_num])
                        ld[ks[-1]].append(img_rect[: self.scale_num])

                    if self.att_flag:
                        img1 = np.concatenate([img1, img_att.reshape(img_att.shape + (1, ))], axis=-1)

                    ld[ks[0]].append(img1)
                    ld[ks[1]].append(img2)

            for b1_source in b1_batch:
                img_handle(['imgs_B1', 'imgs_B1_label', 'imgs_B1_det', 'imgs_B1_rec'],
                           [self.style_B[0], self.style_S],
                           b1_source,
                           is_testing)

            for b2_source in b2_batch:
                img_handle(['imgs_B2', 'imgs_B2_det', 'imgs_B2_rec'],
                           [self.style_B[1]],
                           b2_source,
                           is_testing)

            for s in s_batch:
                img_handle(['imgs_S', 'imgs_S_label', 'imgs_S_det', 'imgs_S_rec'],
                           [self.style_S, self.style_B[0]],
                           s,
                           is_testing)


            for k in ld.keys():
                ld[k] = np.array(ld[k])
                if k.find('rec') == -1:
                    ld[k] = ld[k] / 127.5 - 1

            yield ld

    def imread(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)

    def sub_op(self, img):
        img = np.uint8((img + 1) * 127.5)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img


def sample_images_clear(epoch, batch_i, step, dataloader, g_BS, g_SB, img_log, batch_size=1, conf=None):
    r, c = 3, 4
    if conf.att_loss:
        c = 7

    imgs_dict = dataloader.load_batch_val(batch_size=1)
    imgs_B1, imgs_B1_label= imgs_dict['imgs_B1'], imgs_dict['imgs_B1_label']
    imgs_B2 = imgs_dict['imgs_B2']
    imgs_S, imgs_S_label = imgs_dict['imgs_S'], imgs_dict['imgs_S_label']

    if conf.detail:
        imgs_B1_det, imgs_B1_rec = imgs_dict['imgs_B1_det'], imgs_dict['imgs_B1_rec']
        imgs_B2_det, imgs_B2_rec = imgs_dict['imgs_B2_det'], imgs_dict['imgs_B2_rec']
        imgs_S_det, imgs_S_rec = imgs_dict['imgs_S_det'], imgs_dict['imgs_S_rec']

    if conf.att_loss:
        imgs_B1_in = imgs_B1[:, :, :, : 3]
        imgs_B1_at = imgs_B1[:, :, :, 3:]
        imgs_S_in = imgs_S[:, :, :, : 3]
        imgs_S_at = imgs_S[:, :, :, 3:]

        imgs_B_att = np.concatenate([imgs_B1_at, imgs_B1_at, imgs_B1_at], axis=-1)
        imgs_S_att = np.concatenate([imgs_S_at, imgs_S_at, imgs_S_at], axis=-1)

        imgs_BL_att = ((imgs_B1_in * 0.5 + 0.5) * (imgs_B_att * 0.5 + 0.5) - 0.5) * 2
        imgs_SL_att = ((imgs_S_in * 0.5 + 0.5) * (imgs_S_att * 0.5 + 0.5) - 0.5) * 2

    # model predict
    if conf.detail:
        imgs_B1_in = [imgs_B1_in, imgs_B1_det, imgs_B1_rec]
    fake_S1 = g_BS.predict(imgs_B1_in)
    if conf.att_loss:
        fake_S1_att = ((fake_S1 * 0.5 + 0.5) * (imgs_B_att * 0.5 + 0.5) - 0.5) * 2

    fake_S1_in = fake_S1
    if conf.detail:
        fake_S1_in = [fake_S1_in, imgs_B1_det, imgs_B1_rec]
    reconstr_B1 = g_SB(fake_S1_in)

    if conf.detail:
        fake_S2 = g_BS.predict([imgs_B2, imgs_B2_det, imgs_B2_rec])
        reconstr_B2 = g_SB([fake_S2, imgs_B2_det, imgs_B2_rec])
    else:
        fake_S2 = g_BS.predict(imgs_B2)
        reconstr_B2 = g_SB(fake_S2)
    imgs_B2_label = np.zeros(imgs_B2.shape, dtype=np.uint8)

    if conf.detail:
        fake_B = g_SB.predict([imgs_S_in, imgs_S_det, imgs_S_rec])
        reconstr_S = g_BS([fake_B, imgs_S_det, imgs_S_rec])
    else:
        fake_B = g_SB.predict(imgs_S_in)
        reconstr_S = g_BS(fake_B)

    if conf.att_loss:
        fake_B_att = ((fake_B * 0.5 + 0.5) * (imgs_S_att * 0.5 + 0.5) - 0.5) * 2

    gen_imgs = [imgs_B1_in[0], fake_S1[0], reconstr_B1[0], imgs_B1_label[0], imgs_B_att[0], fake_S1_att[0], imgs_BL_att[0],
                imgs_B2[0], fake_S2[0], reconstr_B2[0], imgs_B2_label[0], imgs_B2_label[0], imgs_B2_label[0], imgs_B2_label[0],
                imgs_S_in[0], fake_B[0], reconstr_S[0], imgs_S_label[0], imgs_S_att[0], fake_B_att[0], imgs_SL_att[0]]

    ms = 256
    cs = 50
    rs = 20
    show_img = np.ones((r * ms + (r - 1) * rs, c * ms + (c - 1) * cs, 3), np.float32)
    cnt = 0
    for i in range(r):
        for j in range(c):
            img = gen_imgs[cnt]
            show_img[i * (ms + rs): i * (ms + rs) + ms,
                     j * (ms + cs): j * (ms + cs) + ms, :] = img * 0.5 + 0.5
            cnt += 1
    show_img = np.uint8(show_img * 255.)
    show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('%s/%d_%d_%d.png' % (img_log, epoch, batch_i, step), show_img)


def val_clear(dataloader, models, valid, fake, conf):
    imgs_dict = dataloader.load_batch_val(batch_size=conf.batch_size)
    imgs_B1, imgs_B1_label = imgs_dict['imgs_B1'], imgs_dict['imgs_B1_label']
    imgs_B2 = imgs_dict['imgs_B2']
    imgs_S, imgs_S_label = imgs_dict['imgs_S'], imgs_dict['imgs_S_label']

    if conf.detail:
        imgs_B1_det, imgs_B1_rec = imgs_dict['imgs_B1_det'], imgs_dict['imgs_B1_rec']
        imgs_B2_det, imgs_B2_rec = imgs_dict['imgs_B2_det'], imgs_dict['imgs_B2_rec']
        imgs_S_det, imgs_S_rec = imgs_dict['imgs_S_det'], imgs_dict['imgs_S_rec']

    hbs = int(conf.batch_size / 2)
    imgs_B = np.concatenate([imgs_B1[: hbs, :, :, : 3], imgs_B2[: hbs]], axis=0)

    if conf.detail:
        imgs_B_det = np.concatenate([imgs_B1_det[: hbs], imgs_B2_det[: hbs]], axis=0)
        imgs_B_rec = np.concatenate([imgs_B1_rec[: hbs], imgs_B2_rec[: hbs]], axis=0)
        fake_S = models['g_BS'].predict([imgs_B, imgs_B_det, imgs_B_rec])
        fake_B = models['g_SB'].predict([imgs_S[:, :, :, : 3], imgs_S_det, imgs_S_rec])
    else:
        fake_S = models['g_BS'].predict(imgs_B)
        fake_B = models['g_SB'].predict(imgs_S[:, :, :, : 3])

    dB_loss_real = models['d_B'].test_on_batch(imgs_B, valid)
    dB_loss_fake = models['d_B'].test_on_batch(fake_B, fake)
    dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

    dS_loss_real = models['d_S'].test_on_batch(imgs_S[:, :, :, : 3], valid)
    dS_loss_fake = models['d_S'].test_on_batch(fake_S, fake)
    dS_loss = 0.5 * np.add(dS_loss_real, dS_loss_fake)

    # Total disciminator loss
    d_loss = 0.5 * np.add(dB_loss, dS_loss)

    # test the generators
    if conf.att_loss:
        imgs_B2_in = np.concatenate([imgs_B2, imgs_B2[:, :, :, 2:]], axis=-1)
        imgs_S_in = imgs_S
        imgs_S = imgs_S[:, :, :, : 3]

    combine_inputs = [imgs_B2_in[: hbs], imgs_S_in[: hbs]]
    if conf.detail:
        combine_inputs += [imgs_B2_det[: hbs], imgs_B2_rec[: hbs],
                           imgs_S_det[: hbs], imgs_S_rec[: hbs]]

    g_loss = models['combined'].test_on_batch(combine_inputs,
                                              [valid[: hbs], valid[: hbs],  # valid_A , valid_B
                                               imgs_B2[: hbs], imgs_S[: hbs],  # reconstr_A , reconstr_B
                                               imgs_B2[: hbs], imgs_S[: hbs],  # img_A_id , img_B_id
                                               ])
    if conf.att_loss:
        imgs_B_att = imgs_B1[: hbs, :, :, 3:]
        imgs_S_att = imgs_S_in[:, :, :, 3:]
        imgs_B_att = np.concatenate([imgs_B_att, imgs_B_att, imgs_B_att], axis=-1)
        imgs_S_att = np.concatenate([imgs_S_att, imgs_S_att, imgs_S_att], axis=-1)
        att_BL = ((imgs_B1_label[: hbs] * 0.5 + 0.5) * (imgs_B_att * 0.5 + 0.5) - 0.5) * 2
        att_SL = ((imgs_S_label * 0.5 + 0.5) * (imgs_S_att * 0.5 + 0.5) - 0.5) * 2

        imgs_B1_in = imgs_B1
        imgs_B1 = imgs_B1[:, :, :, : 3]

    combine_l_inputs = [imgs_B1_in[: hbs], imgs_S_in[hbs:]]
    if conf.detail:
        combine_l_inputs += [imgs_B1_det[: hbs], imgs_B1_rec[hbs:],
                             imgs_S_det[: hbs], imgs_S_rec[hbs:]]

    g_loss_l = models['combined_l'].test_on_batch(combine_l_inputs,
                                            [valid[: hbs], valid[hbs:],
                                             imgs_B1[: hbs], imgs_S[hbs:],
                                             imgs_B1[: hbs], imgs_S[hbs:],
                                             imgs_B1_label[: hbs], imgs_S_label[hbs:],
                                             att_BL, att_SL[: hbs]])
    g_loss[0: 7] = np.array(g_loss_l[0: 7] + g_loss[0: 7]) / 2


    return d_loss, g_loss


if __name__ == '__main__':

    from configs.cf_clear_stage2 import config as conf
    dl = DataLoader(conf)

    # for batch_i, imgs_dict in enumerate(dl.load_batch_val(conf.batch_size)):
    imgs_dict = dl.load_batch_val(conf.batch_size)
    if imgs_dict is not None:

        items = ['imgs_B1', 'imgs_B2', 'imgs_S']
        for item in items:
            print(item)
            img_B1 = imgs_dict[item][0]
            img_B1, img_B1_att = img_B1[:, :, : 3], img_B1[:, :, 3:]
            img_B1 = dl.sub_op(img_B1)

            if '%s_label' % item in imgs_dict.keys():
                img_B1_att = np.concatenate([img_B1_att, img_B1_att, img_B1_att], axis=-1)
                img_B1_att = dl.sub_op(img_B1_att)

                img_B1_label = imgs_dict['%s_label' % item][0]
                img_B1_label = dl.sub_op(img_B1_label)

            img_B1_det = imgs_dict['%s_det' % item][0]
            img_B1_dets = []
            for i in range(conf.scale_num):
                t = img_B1_det[i, :, :, :]
                img_B1_dets.append(dl.sub_op(t))

            if '%s_label' % item in imgs_dict.keys():
                combine = cv2.hconcat((img_B1, img_B1_att, img_B1_label, img_B1_dets[0], img_B1_dets[1], img_B1_dets[2]))
            else:
                combine = cv2.hconcat((img_B1, img_B1_dets[0], img_B1_dets[1], img_B1_dets[2]))
            cv2.imshow('img', combine)
            cv2.waitKey()




