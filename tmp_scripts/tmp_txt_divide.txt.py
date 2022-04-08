import cv2
import numpy as np
import os
from libs.datautils import get_binimg
from tqdm import tqdm

def lap_img(img, ):
    d = 16
    l_img = np.zeros(img.shape, np.float32)
    r, c = int(img.shape[0] / d), int(img.shape[1] / d)
    for i in range(r):
        for j in range(c):
            tmp_img = img[i * d: (i + 1) * d, j * d: (j + 1) * d]
            l_img[i * d: (i + 1) * d, j * d: (j + 1) * d] = cv2.Laplacian(tmp_img, cv2.CV_64F).var()
    return l_img



def tmp():
    data_path = 'D:/paper_stage2'

    items = [
            # '3D_to_3D',
            'our_to_3D',
    ]

    for item in items:
        names = open(os.path.join(data_path, 'data_r/txt/our.txt'), 'r').readlines()
        names = [name.strip() for name in names]
        print(len(names))

        save_path = os.path.join(data_path, 'data_r/%s_bin' % item)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for name in tqdm(names):
            img = cv2.imread(os.path.join(data_path, 'data_r', item, name))
            bin_img = get_binimg(img)
            bin_img = np.uint8(bin_img * 255.)
            bin_img = np.stack([bin_img, bin_img, bin_img], axis=-1)
            cv2.imwrite(os.path.join(save_path, name), bin_img)

def t():

    import cv2

    img_path = r'D:\20x_and_40x_data\split_data\10140015_10000_13824_0.tif'
    img = cv2.imread(img_path)

    for i in range(0, 100):
        img = cv2.GaussianBlur(img, (13, 13), sigmaX=2, sigmaY=2)

    cv2.imwrite('t.png', img)


if __name__ == '__main__':
    pass