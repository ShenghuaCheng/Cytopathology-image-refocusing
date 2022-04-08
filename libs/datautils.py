

import numpy as np
import cv2

def pre_op(img):
    '''
    :param img: img = cv2.imread()
    :return: g_BS model input
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img) / 127.5 - 1
    return img


def suf_op(img):
    '''
    :param img: g_BS model output
    :return: img = cv2.imwrite()
    '''
    img = np.uint8((img + 1) * 127.5)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


from scipy import ndimage as ndi
from skimage import morphology

def get_binimg(img , col_thre=16, vol_thre=900):
    '''
    :param img:
    :param col_thre:
    :param vol_thre:
    :return: bool type matrix
    '''
    wj1 = img.max(axis=2)
    wj2 = img.min(axis=2)
    wj3 = wj1 - wj2
    img_bin = wj3 > col_thre
    img_bin = morphology.remove_small_objects(img_bin, min_size=vol_thre)
    img_bin = ndi.binary_fill_holes(img_bin > 0)
    return img_bin