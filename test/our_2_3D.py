'''
change img style
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from configs.cf_style_gray_rb import config as conf
from mmodels.models import build_generator_style

import datetime
import cv2
import numpy as np
from tqdm import tqdm

g_BA = build_generator_style(conf)

pre_path = 'D:/paper_stage2/data_r'
item = 'our'
data_path = os.path.join(pre_path, item)
#test txt path
txt_path = os.path.join(pre_path, 'txt/%s.txt' % item)
#img save path
save_path = os.path.join(pre_path, '%s_to_3D' % item)

def pre_op(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img) / 127.5 - 1
    return img

def suf_op(img):
    img = np.uint8((img + 1) * 127.5)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

if __name__ == '__main__':

    start_time = datetime.datetime.now()

    path = 'D:/paper_stage2/log'
    # load weights
    weights = [
        'style/wei_log/g_BA_14.h5',
        'style_gray/wei_log/g_our_3D_12_12_67500.h5',
        'style_rb/wei_log/g_our_3D_23_1940_105000.h5',
        'style_gray_rb/wei_log/g_our_3D_22_1122_56100.h5',
    ]

    name_list = []
    with open(txt_path, 'r') as txt_list:
        for line in txt_list:
            line = line.strip()
            name_list.append(line)

    # random.shuffle(name_list)
    nums = 2000
    for weight in weights:
        print(weight)
        weight_path = os.path.join(path, weight)
        g_BA.load_weights(weight_path)

        save_path = os.path.join(path, weight.split('/')[0], 'test')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for line in tqdm(name_list[: nums]):
            line = line.strip()
            img_path = os.path.join(data_path, line)
            img = cv2.imread(img_path)
            img = pre_op(img)
            img_B = np.reshape(img, (1, 256, 256, 3))
            fake_A = g_BA.predict(img_B)
            fake_A = fake_A[0]
            fake_A = suf_op(fake_A)

            cv2.imwrite('%s/%s' % (save_path, line), fake_A)

        elapsed_time = datetime.datetime.now() - start_time


    combine_path = os.path.join(path, 'tmp')
    if not os.path.exists(combine_path):
        os.makedirs(combine_path)
    for line in tqdm(name_list[: nums]):

        imgs = []
        our = cv2.imread(os.path.join(data_path, line))
        imgs.append(our)

        for weight in weights:
            uints = weight.split('/')
            img = cv2.imread(os.path.join(path, uints[0], 'test', line))
            imgs.append(img)

        combine_img = cv2.hconcat(imgs)
        cv2.imwrite(os.path.join(combine_path, line), combine_img)
