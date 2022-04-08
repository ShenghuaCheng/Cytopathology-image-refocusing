import numpy as np
import cv2
import os
import random

def rb(x, channel = 0):
    x = x / 255.
    exp_x = np.exp(x)
    sum_x = np.sum(exp_x, axis=-1)
    sum_x = np.stack([sum_x, sum_x, sum_x], axis=-1)
    x = np.divide(exp_x, sum_x)
    x1 = np.equal(np.max(x, axis=2), x[:, :, channel])
    x3 = np.multiply(x1, x[:, :, channel])
    x4 = np.multiply(np.subtract(x3, 1. / 3), 3. / 2)

    x4[x4 < 0] = 0
    x4[x4 > 1] = 1
    x6 = np.multiply(x4, 1 / (np.max(x4) - np.min(x4)))
    x6[x6 < 0] = 0
    x6[x6 > 1] = 1
    return np.uint8(x6 * 255)

save_path = r'D:\paper_stage2\data_r\tmp'

item = 'our'
path = r'D:\paper_stage2\data_r\%s' % item
names = os.listdir(path)
random.shuffle(names)
for name in names[:300]:
    name = name[: name.find('.')]
    img_path = '%s/%s.png' % (path, name)
    x = cv2.imread(img_path)
    r_img = rb(x, 2)
    b_img = rb(x, 0)

    r_img = np.stack([r_img * 0, r_img * 0, r_img], axis=-1)
    b_img = np.stack([b_img , b_img * 0, b_img * 0], axis=-1)

    cv2.imwrite('%s/%s.png' % (save_path, name), x)
    cv2.imwrite('%s/%s_r.png' % (save_path, name), r_img)
    cv2.imwrite('%s/%s_b.png' % (save_path, name), b_img)
