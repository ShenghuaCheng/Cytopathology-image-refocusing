
import cv2
import numpy as np
import os
from tqdm import tqdm

from libs.kernel_seg import KernelSeg

class IOU:


    def __init__(self):
        pass


    def __read_contours__(self, txt_path):
        lines = open(txt_path, 'r').readlines()
        contours = []
        for line in lines:
            line = line[1: -2]
            points = line.split('Point:')
            points = points[1:]
            final_points = []
            for point in points:
                us = point.split(',')
                us = us[: 2]
                us[0] = int(float(us[0].lstrip().strip()))
                us[1] = int(float(us[1].lstrip().strip()))
                final_points.append(us)
            contours.append(final_points)
        return contours

    def __save_label__(self, contours, path):

        img = np.zeros((256, 256, 3), dtype=np.uint8)
        for contour in contours:
            img = cv2.fillPoly(img, np.array([contour]), (255, 255, 255))
        cv2.imwrite(path, img)


    def __iou__(self, predict, label):
        predict = predict > 0
        label = label > 0
        numerator = predict * label
        denominator = predict + label
        iou = np.sum(numerator) / np.sum(denominator)
        return iou

    def __rp__(self, x, y):
        contours, _ = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        y = y > 0
        recall = 0
        for contour in contours:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            c = np.array(contour, dtype=np.int)[:, 0, :]
            img = cv2.fillPoly(img, [c], (255, 255, 255))
            img = img > 0
            img = img[:, :, 0]

            inter_area = np.sum(img * y)
            area = np.sum(img)
            if inter_area / area > 0.7:
                recall = recall + 1

        recall = 0 if len(contours) == 0 else recall / len(contours)
        return recall

    def __recall__(self, predict, label): return self.__rp__(label, predict)

    def __precision__(self, predict, label): return self.__rp__(predict, label)

path = 'D:/paper_stage2/data_r'

# txt to label img
def txt2label():
    iou = IOU()
    item = 'our_to_3D'
    txts = os.listdir(os.path.join(path, '%s_test/project/scripts' % item))
    txts = [x for x in txts if x.find('txt') != -1]

    for txt in txts:
        contours = iou.__read_contours__(os.path.join(path, '%s_test/project/scripts' % item, txt))
        if len(contours) > 0:
            name = txt[: txt.find('.txt')]
            save_path = os.path.join(path, '%s_test/project/labels' % item, name + '.png')
            iou.__save_label__(contours, save_path)


def test_kernel_seg(item='our_to_3D'):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    ks = KernelSeg()
    iu = IOU()

    label_path = os.path.join(path, '%s_test/project/labels' % item)
    names = os.listdir(label_path)
    names = [x for x in names if x.find('.png') != -1]

    item = 'data_r/our_to_3D'
    inputs_path = os.path.join(r'D:\paper_stage2', item)

    save_path = r'D:\paper_stage2\tmp'

    ious, recalls, precisions = [], [], []
    for name in tqdm(names):
        input_path = os.path.join(inputs_path, name)
        img = cv2.imread(input_path)
        res = ks.predict_path(input_path)

        label = cv2.imread(os.path.join(label_path, name), 0)
        iou = iu.__iou__(res, label)
        recall = iu.__recall__(res, label)
        precision = iu.__precision__(res, label)
        if True:
            item = item.replace('/', '_').replace('\\', '_')
            s_path = os.path.join(save_path, item)
            if not os.path.exists(s_path):
                os.makedirs(s_path)
            # with open('%s/recall.txt' % s_path, 'w+') as txt:
            #     txt.write('%s:%.6f\n' % (name, recall))

            contours, _ = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cimg = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(s_path, 'c_' + name), cimg)
            cv2.imwrite(os.path.join(s_path, 'r_' + name), img)

            contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            limg = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(s_path, 'l_' + name), limg)

        # print(iou)
        ious.append(iou)
        recalls.append(recall)
        precisions.append(precision)
    print('mean iou: %.4f, mean recalll: %.4f, mean precision: %.4f'%
          (np.mean(iou), np.mean(recalls), np.mean(precisions)))


def concate_imgs():

    item = 'our_to_3D'
    # list kernel label imgs name
    label_path = os.path.join(path, '%s_test/project/labels' % item)
    names = os.listdir(label_path)
    names = [x for x in names if x.find('.png') != -1]

    folder_path = r'D:\paper_stage2\tmp'
    folders = [
        'data_r_our_to_3D',
        'log_clear_stage1(att_loss&no_rb)_test_our_to_3D',
        'log_clear_stage2(att_loss&no_rb)_test_our_to_3D',
    ]

    save_path = os.path.join(folder_path, 'concate_%s' % item)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for name in names:

        r_imgs, c_imgs = [], []

        for folder in folders:

            r_img = cv2.imread(os.path.join(folder_path, folder, 'r_' + name))
            c_img = cv2.imread(os.path.join(folder_path, folder, 'c_' + name))

            r_imgs.append(r_img)
            c_imgs.append(c_img)

        r_img = cv2.hconcat(r_imgs)
        c_img = cv2.hconcat(c_imgs)

        concate_img = cv2.vconcat((r_img, c_img))

        cv2.imwrite(os.path.join(save_path, name), concate_img)


if __name__ == '__main__':
    # test_kernel_seg()
    # txt2label()
    concate_imgs()