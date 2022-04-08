import cv2
from openslide import OpenSlide
import openslide
import numpy as np
import os


path1 = r'E:\XYW-MFP\1910230038.mrxs'
path2 = r'E:\XYW-MFP\multi-layer\1910230038\1910230038_Wholeslide_Extended.tif'
ors1 = OpenSlide(path1)
img1 = ors1.read_region((0,0), 5, ors1.level_dimensions[5])
img1 = np.array(img1)
img1 = cv2.cvtColor(img1[:,:,0:3], cv2.COLOR_BGR2GRAY)
ors2 = OpenSlide(path2)
img2 = ors2.read_region((20000,20000), 0, (3200,3200))
img2 = np.array(img2)
img2 = cv2.cvtColor(img2[:,:,0:3], cv2.COLOR_BGR2GRAY)
img2 = cv2.resize(img2, (100,100))

cv2.imwrite(r'E:\20x_and_40x_new\img1.jpg', img1)
cv2.imwrite(r'E:\20x_and_40x_new\img2.jpg', img2)
res = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
w, h = img2.shape[::-1]
loc = np.where(res >= (res.max()-1e-9))
pt = (loc[0][0], loc[1][0])
cv2.rectangle(img1, pt, (pt[0] + w, pt[1] + h), (0, 0, 0),2)
cv2.imwrite(r'E:\20x_and_40x_new\img3.jpg', img1)
offset = (pt[0]*32 - 20000, pt[1]*32 - 20000)

img3 =ors1.read_region((40000+offset[1],40000+offset[0]), 0, (3024,3024))
img3 = np.array(img3)
img3 = img3[:,:,0:3]
img4 =ors2.read_region((40000,40000), 0, (3024,3024))
img4 = np.array(img4)
img4 = img4[:,:,0:3]
cv2.imwrite(r'E:\20x_and_40x_new\img4.jpg', img3[:,:,::-1])
cv2.imwrite(r'E:\20x_and_40x_new\img5.jpg', img4[:,:,::-1])

level = 0
block_size = 10000
ors1 = OpenSlide(path1)
wmin = int(ors1.properties[openslide.PROPERTY_NAME_BOUNDS_X])
hmin = int(ors1.properties[openslide.PROPERTY_NAME_BOUNDS_Y])
position = (13344-500+wmin,64864-500+hmin)
img1 = ors1.read_region(position , level , (block_size , block_size))
img1 = np.array(img1)
img1 = img1[:,:,0:3]
cv2.imwrite(r'E:\20x_and_40x_new\img1.jpg', img1[:,:,::-1])
ors2 = OpenSlide(path2)
img2 = ors2.read_region((10000,20000) , level , (block_size , block_size))
img2 = np.array(img2)
img2 = img2[:,:,0:3]
cv2.imwrite(r'E:\20x_and_40x_new\img2.tif', img2[:,:,::-1])


def read_txt(file_path):
    ptx = []
    pty = []
    with open(file_path) as f:
        for line in f:
            line = line[: -1]
            line_uints = line.split(',')
            ptx.append(int(line_uints[1]))
            pty.append(int(line_uints[2]))
    return ptx,pty
path1 = r'E:\XYW-MFP\1910230038.mrxs'
path2_1 = r'E:\XYW-MFP\multi-layer\1910230038\1910230038_Wholeslide_0.tif'
path2_2 = r'E:\XYW-MFP\multi-layer\1910230038\1910230038_Wholeslide_-2.tif'
path2_3 = r'E:\XYW-MFP\multi-layer\1910230038\1910230038_Wholeslide_2.tif'
path2_4 = r'E:\XYW-MFP\multi-layer\1910230038\1910230038_Wholeslide_-4.tif'
path2_5 = r'E:\XYW-MFP\multi-layer\1910230038\1910230038_Wholeslide_4.tif'
path2_6 = r'E:\XYW-MFP\multi-layer\1910230038\1910230038_Wholeslide_Extended.tif'
path2_list = [path2_1,path2_2,path2_3,path2_4,path2_5,path2_6]
name_list = ['0','-2','2','-4','4', 'Extended']
file_path = r'F:\SoftWareCytologyAI\SlideCalc\Result\refocus1\1910230038.txt'
path_save_1 = r'E://XYW-MFP-Patches//lesion_cell//1910230038//'
path_save_2 = r'E://XYW-MFP-Patches//lesion_cell_review\1910230038//'
offset = (57600, 256)
size_1 = 305
size_2 = 1207
n = 200
ors1 = OpenSlide(path1)
wmin = int(ors1.properties[openslide.PROPERTY_NAME_BOUNDS_X])
hmin = int(ors1.properties[openslide.PROPERTY_NAME_BOUNDS_Y])
ptx,pty = read_txt(file_path)

for i in range(n):
    position = (ptx[i]-int(size_1/2)+wmin, pty[i]-int(size_1/2)+hmin)
    img = ors1.read_region(position , 0, (size_1, size_1))
    img = np.array(img)
    img = img[:, :, 0:3]
    img = cv2.resize(img, (256, 256))
    cv2.imwrite(path_save_1+np.str(i)+'_fu.tif', img[:,:,::-1])
    position = (ptx[i] - int(size_2 / 2) + wmin, pty[i] - int(size_2 / 2) + hmin)
    img = ors1.read_region(position, 0, (size_2, size_2))
    img = np.array(img)
    img = img[:, :, 0:3]
    img = cv2.resize(img, (1000, 1000))
    cv2.imwrite(path_save_2 + np.str(i) + '_fu.tif', img[:, :, ::-1])
    for j in range(6):
        path2_temp = path2_list[j]
        ors2 = OpenSlide(path2_temp)
        position = (ptx[i] - int(size_1 / 2) + wmin-offset[1], pty[i] - int(size_1 / 2) + hmin-offset[0])
        img = ors2.read_region(position, 0, (size_1, size_1))
        img = np.array(img)
        img = img[:, :, 0:3]
        img = cv2.resize(img, (256, 256))
        cv2.imwrite(path_save_1 + np.str(i) + '_'+name_list[j]+'.tif', img[:, :, ::-1])
        position = (ptx[i] - int(size_2 / 2) + wmin-offset[1], pty[i] - int(size_2 / 2) + hmin-offset[0])
        img = ors2.read_region(position, 0, (size_2, size_2))
        img = np.array(img)
        img = img[:, :, 0:3]
        img = cv2.resize(img, (1000, 1000))
        cv2.imwrite(path_save_2 + np.str(i) + '_'+name_list[j]+'.tif', img[:, :, ::-1])

path1 = r'E:\20x_and_40x_new\20x_mrxs\10140074.mrxs'
path2_1 = r'E:\20x_and_40x_new\20x_tiff\10140074\10140074_Wholeslide_0.tif'
path2_2 = r'E:\20x_and_40x_new\20x_tiff\10140074\10140074_Wholeslide_-1.tif'
path2_3 = r'E:\20x_and_40x_new\20x_tiff\10140074\10140074_Wholeslide_1.tif'
path2_4 = r'E:\20x_and_40x_new\20x_tiff\10140074\10140074_Wholeslide_-2.tif'
path2_5 = r'E:\20x_and_40x_new\20x_tiff\10140074\10140074_Wholeslide_2.tif'
path2_6 = r'E:\20x_and_40x_new\20x_tiff\10140074\10140074_Wholeslide_-3.tif'
path2_7 = r'E:\20x_and_40x_new\20x_tiff\10140074\10140074_Wholeslide_3.tif'
path2_8 = r'E:\20x_and_40x_new\20x_tiff\10140074\10140074_Wholeslide_Extended.tif'
path2_list = [path2_1,path2_2,path2_3,path2_4,path2_5,path2_6,path2_7,path2_8]
name_list = ['0','-1','1','-2','2', '-3','3', 'Extended']
file_path = r'F:\SoftWareCytologyAI\SlideCalc\Result\refocus\10140074.txt'
path_save_1 = r'E://XYW-MFP-Patches//lesion_cell//10140074//'
path_save_2 = r'E://XYW-MFP-Patches//lesion_cell_review\10140074//'
offset = (47872, 0)
size_1 = 256
size_2 = 1000
n = 200
ors1 = OpenSlide(path1)
wmin = int(ors1.properties[openslide.PROPERTY_NAME_BOUNDS_X])
hmin = int(ors1.properties[openslide.PROPERTY_NAME_BOUNDS_Y])
ptx,pty = read_txt(file_path)

for i in range(n):
    position = (ptx[i]-int(size_1/2)+wmin, pty[i]-int(size_1/2)+hmin)
    img = ors1.read_region(position , 0, (size_1, size_1))
    img = np.array(img)
    img = img[:, :, 0:3]
    cv2.imwrite(path_save_1+np.str(i)+'_fu.tif', img[:,:,::-1])
    position = (ptx[i] - int(size_2 / 2) + wmin, pty[i] - int(size_2 / 2) + hmin)
    img = ors1.read_region(position, 0, (size_2, size_2))
    img = np.array(img)
    img = img[:, :, 0:3]
    cv2.imwrite(path_save_2 + np.str(i) + '_fu.tif', img[:, :, ::-1])
    for j in range(8):
        path2_temp = path2_list[j]
        ors2 = OpenSlide(path2_temp)
        position = (ptx[i] - int(size_1 / 2) + wmin-offset[1], pty[i] - int(size_1 / 2) + hmin-offset[0])
        img = ors2.read_region(position, 0, (size_1, size_1))
        img = np.array(img)
        img = img[:, :, 0:3]
        img = cv2.resize(img, (256, 256))
        cv2.imwrite(path_save_1 + np.str(i) + '_'+name_list[j]+'.tif', img[:, :, ::-1])
        position = (ptx[i] - int(size_2 / 2) + wmin-offset[1], pty[i] - int(size_2 / 2) + hmin-offset[0])
        img = ors2.read_region(position, 0, (size_2, size_2))
        img = np.array(img)
        img = img[:, :, 0:3]
        img = cv2.resize(img, (1000, 1000))
        cv2.imwrite(path_save_2 + np.str(i) + '_'+name_list[j]+'.tif', img[:, :, ::-1])

file_list = os.listdir(r'E:\XYW-MFP-Patches\lesion_cell\1910230038')
file_path = r'E:\XYW-MFP-Patches\lesion_cell\1910230038.txt'
file_list = [temp for temp in file_list if '.tif' in temp]
def write_txt(list, file_path):
    f =  open(file_path, 'w')
    for temp in list:
        temp1 = r'E:\XYW-MFP-Patches\lesion_cell\1910230038' + '\\' + temp + '\n'
        f.write(temp1)
    f.close()
    return None
write_txt(file_list, file_path)
