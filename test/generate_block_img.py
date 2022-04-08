import openslide
import cv2
import numpy as np


def read_region(ors, location, level, size):
    img = ors.read_region(location, level, size)
    img = np.array(img)
    img = img[:, :, : 3]
    img = img[:, :, ::-1]
    return img

slide_path = r'D:\paper_stage2\data_r\WNLO_Sfy3_pos\1162052.svs'

tmp_path = r'E:\tmp'
b_w = 1600
b_h = 640

ors = openslide.OpenSlide(slide_path)

s = ors.level_dimensions[0]
print(s)

s_w = 1000
s_h = 0


for i in range(100):
    for j in range(100):
        img = read_region(ors, (s_w + i * b_w, s_h + j * b_h), 0, (b_w, b_h))
        cv2.imwrite('%s/%d_%d.jpg' % (tmp_path, s_w + i * b_w, s_h + j * b_h), img)

ors.close()