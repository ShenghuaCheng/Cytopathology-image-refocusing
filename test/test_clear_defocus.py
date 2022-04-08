'''
test img clear model
'''
from test.cfs_test.cf_t_clear import config as conf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpus

from mmodels.mb import g_nets_dict
from dls.dl_tools import __get_binimg__
from test.test_tools import pre_op, suf_op
from libs.metrics import cal_metrics

import datetime
import cv2
import numpy as np

# Build the generators
g_BS = g_nets_dict[conf.g_net](conf)

if not os.path.exists(conf.save_path):
    os.makedirs(conf.save_path)


def get_defocus_rect(blur_img_att):
    maxv = np.max(blur_img_att)
    max_bin_blur = (blur_img_att == maxv) * blur_img_att
    max_bin_blur = np.uint8(max_bin_blur)
    contours, _ = cv2.findContours(max_bin_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = 0, 0, 0, 0
    for contour in contours:
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        if w1 > w and h1 > h:
            x, y, w, h = x1, y1, w1, h1
    return x, y, w, h

if __name__ == '__main__':

    name_list = list()
    tmp = open(conf.txt_path, 'r')
    for line in tmp:
        line = line.strip()
        name_list.append(line)
    tmp.close()

    # chhose imgs
    test_imgs = name_list
    test_imgs = name_list[: conf.test_nums]

    for model_index, test_model in enumerate(conf.test_models):
        weight_path = os.path.join(conf.weig_path, test_model)
        g_BS.load_weights(weight_path)

        start_time = datetime.datetime.now()

        metrics_dict = {}
        for me in conf.metrics:
            metrics_dict[me] = []

        for k, name in enumerate(test_imgs):
            print(k, name)
            img = cv2.imread(os.path.join(conf.x_path, name))
            blur_img = pre_op(img)
            blur_img = np.reshape(blur_img, (1, ) + blur_img.shape)

            blur_img_att = cv2.imread(os.path.join(conf.att_path, name[: name.find('.png')] + '_PC.png'))
            blur_img_att = np.mean(blur_img_att, axis=-1)
            blur_img_att = blur_img_att * __get_binimg__(img, conf.col_thre, conf.vol_thre)
            x, y, w, h = get_defocus_rect(blur_img_att)

            if conf.att_flag:             # if att_flag, add att_map
                blur_img_att = np.float32(blur_img_att) / 127.5 - 1.
                blur_img_att = np.reshape(blur_img_att, (1, ) + blur_img_att.shape + (1, ))
                blur_img = np.concatenate([blur_img, blur_img_att], axis=-1)

            gen_img = g_BS.predict(blur_img)[0]
            gen_img = suf_op(gen_img)
            label = cv2.imread(os.path.join(conf.y_path, name))

            if w > 10 and h > 10:
                img = img[y: y + h, x: x + w, :]
                gen_img = gen_img[y: y + h, x: x + w, :]
                label = label[y: y + h, x: x + w, :]
            print(np.shape(img))
            re_dict = cal_metrics(gen_img, label, choose_metrics=conf.metrics)
            for me in conf.metrics:
                metrics_dict[me].append(re_dict[me])

            combined = cv2.hconcat([img, gen_img])
            if label is not None:
                combined = cv2.hconcat([combined, label])
            cv2.imwrite(os.path.join(conf.save_path, name), combined)

        elapsed_time = datetime.datetime.now() - start_time
        print(weight_path)
        for me in conf.metrics:
            print('%s: %.4f, ' % (me, np.mean(metrics_dict[me])), end='')
        print('time consumption: %s' % elapsed_time)
