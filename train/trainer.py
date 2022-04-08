import cv2
import numpy as np
import os
import random

class Trainer:

    def __init__(self):
        pass


    def __find_fore_rects1__(self, path):
        img = cv2.imread(path)
        if img is None:
            raise Exception('%s can not read!' % path)
        return self.__find_fore_rects__(img)

    # get random fore points:
    def grfp(self, data_path, item, name, nums=3):
        ker_bin = cv2.imread(os.path.join(data_path, '%s_kernel' % item, name), 0)
        for_bin = cv2.imread(os.path.join(data_path, '%s_bin' % item, name), 0)

        ker_bin = (1 - ker_bin) > 0
        for_bin = for_bin > 0
        for_bin = for_bin * ker_bin

        non_zero_points = np.flatnonzero(for_bin)
        random.shuffle(non_zero_points)
        ps = []
        for nzp in non_zero_points[: nums]:
            ps.append([int(nzp / 256), nzp % 256])

        return ps

    def __find_fore_rects__(self, img):
        assert img is not None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rects = []
        for contour in contours:
            rect = cv2.boundingRect(contour)
            rects.append(rect)
        return rects

    def block(self, img, data_path, item, img_name, nums=3):
        kernel_seg_path = os.path.join(data_path, '%s_kernel' % item, img_name)
        kernel_rects = trainer.__find_fore_rects1__(kernel_seg_path)
        kernel_rects = sorted(kernel_rects, key=lambda x: x[2] * x[3])
        if len(kernel_rects) < nums:
            kernel_rects += ([[0, 0, 0, 0]] * nums)

        fore_path = os.path.join(data_path, '3D_to_3D_bin', img_name)
        fore_rects = trainer.__find_fore_rects1__(fore_path)
        if len(fore_rects) < nums:
            fore_rects += ([[0, 0, 0, 0]] * nums)

        rects = kernel_rects[: nums] + fore_rects[: nums]
        block_imgs, block_rects = [], []
        for ind, r in enumerate(rects):
            if r != [0, 0, 0, 0]:
                kf = img[r[1]: r[1] + r[3], r[0]: r[0] + r[2], :]
                kf = cv2.resize(kf, (256, 256))
                block_rects.append([1, r[0], r[1], r[2], r[3]])
            else:
                kf = np.zeros((256, 256, 3))
                block_rects.append([0, 0, 0, 0, 0])
            block_imgs.append(kf)

        block_rects = np.array(block_rects)
        block_imgs = np.array(block_imgs)
        return block_imgs, block_rects

    def block1(self, img, data_path, item, img_name, nums=3):
        kernel_seg_path = os.path.join(data_path, '%s_kernel' % item, img_name)
        kernel_rects = trainer.__find_fore_rects1__(kernel_seg_path)
        kernel_rects = sorted(kernel_rects, key=lambda x: x[2] * x[3], reverse=True)
        if len(kernel_rects) < nums:
            kernel_rects += ([[0, 0, 0, 0]] * nums)

        s = 32
        # handle rect
        for ind in range(nums):
            if kernel_rects[ind] != [0, 0, 0, 0]:
                r = kernel_rects[ind]
                cx, cy = r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)
                kernel_rects[ind] = [cx - s, cy - s, s * 2, s * 2]

        for_points = trainer.grfp(data_path, item, img_name, nums)
        if len(for_points) < nums:
            for_points += ([[0, 0]] * nums)

        for_rects = []
        for fp in for_points:
            if fp != [0, 0]:
                for_rects.append([fp[0] - s, fp[1] - s, s * 2, s * 2])
            else:
                for_rects.append([0, 0, 0, 0])

        rects = kernel_rects[: nums] + for_rects[: nums]
        block_imgs, block_rects = [], []
        img = cv2.copyMakeBorder(img, s * 2, s * 2, s * 2, s * 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for ind, r in enumerate(rects):
            if r != [0, 0, 0, 0]:
                r[0] = r[0] + s * 2
                r[1] = r[1] + s * 2
                kf = img[r[1]: r[1] + r[3], r[0]: r[0] + r[2], :]
                kf = cv2.resize(kf, (256, 256))
                block_rects.append([1, r[0], r[1], r[2], r[3]])
            else:
                kf = np.zeros((256, 256, 3))
                block_rects.append([0, 0, 0, 0, 0])
            block_imgs.append(kf)

        block_rects = np.array(block_rects)
        block_imgs = np.array(block_imgs)
        return block_imgs, block_rects


    def find_kernels(self, img, data_path, item, name, s=32):
        kernel_seg_path = os.path.join(data_path, '%s_kernel' % item, name)
        kernel_rects = trainer.__find_fore_rects1__(kernel_seg_path)
        kernel_rects = sorted(kernel_rects, key=lambda x: x[2] * x[3], reverse=True)

        # handle rect
        for ind in range(len(kernel_rects)):
            r = kernel_rects[ind]
            cx, cy = r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)
            kernel_rects[ind] = [cx - s, cy - s, s * 2, s * 2]

        ker_imgs, ker_rects = [], []
        img = cv2.copyMakeBorder(img, s * 2, s * 2, s * 2, s * 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for ind, r in enumerate(kernel_rects):
            r[0] = r[0] + s * 2
            r[1] = r[1] + s * 2
            ker_img = img[r[1]: r[1] + r[3], r[0]: r[0] + r[2], :]
            ker_rects.append([1, r[0], r[1], r[2], r[3]])
            ker_imgs.append(ker_img)

        ker_rects = np.array(ker_rects)
        ker_imgs = np.array(ker_imgs)
        return ker_imgs, ker_rects

trainer = Trainer()

if __name__ == '__main__':

    data_path = r'D:\paper_stage2\data_r'
    img_name = '10140015_6468_36960.png'
    item = '3D_to_3D'

    trainer = Trainer()

    img = cv2.imread(os.path.join(data_path, item, img_name))

    ker_imgs, ker_rects = trainer.find_kernels(img, data_path, item, img_name)



