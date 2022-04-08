import cv2
import os

class CropImg:

    def __init__(self):

        self.data_path = 'D:/paper_stage2'

        self.items = {
            '3D_to_3D': {
                # '10140015_1232_34804.png': [120, 145, 100, 100],
                # '10140015_1540_25256.png': [155, 17, 100, 100],
                # '10140015_1540_34804.png': [100, 115, 100, 100],
                # '10140015_36960_66528.png': [134, 54, 100, 100]
            },
            # 10140015_36960_66528
            'our_to_3D': {
                # '1162034_23552_22528.png': [140, 145, 100, 100],
                # '1162034_23808_64256.png': [155, 60, 100, 100],
                # '1162034_32000_45312.png': [119, 17, 100, 100],
                # '1162034_20992_57344.png': [110, 60, 100, 100],

                # show whole slide image
                '1162034_1280_21760.png':    [148 + 25, 25 + 25, 50, 50],
                '1162034_13824_11520.png':  [171 + 25, 78 + 25, 50, 50],
                '1162034_13824_64768.png':  [80 + 25, 11 + 25, 50, 50]
            }
        }

        self.cp_items = [
                    # 'cp_imgs/SIUN',
                    # 'cp_imgs/DeblurGAN',
                    # 'cp_imgs/DeblurGANv2',
                    # 'cp_imgs/DeepDeblur-PyTorch',
                    # 'cp_imgs/DMPHN-cvpr19-master',
                    # 'cp_imgs/CycleGAN',
                    # 'cp_imgs/Unsupervised-Domain-Specific-Deblurring',
                    # 'clear_stage1(att_loss&no_rb)/test',
                    # 'cp_imgs/SRN'
                    #     'clear_stage2(att_loss&no_rb)/test'
                    ]

        self.save_path = 'D:/paper_stage2/visio/cp_imgs/stage2/our_to_3D1'


    def __crop_img__(self, img, coor, name, mk):
        croped_img = img[coor[0]: coor[0] + coor[2], coor[1]: coor[1] + coor[3], :]
        croped_img = cv2.resize(croped_img, img.shape[: 2])

        img = cv2.rectangle(img, (coor[1], coor[0]), (coor[1] + coor[3], coor[0] + coor[2]), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(self.save_path, 'big_%s_%s' % (mk, name)), img)
        cv2.imwrite(os.path.join(self.save_path, 'sma_%s_%s' % (mk, name)), croped_img)

    def crop_items(self, item='3D_to_3D'):

        items = self.items[item]
        for name, coor in items.items():
            print(name, coor)

            defocused_img = cv2.imread(os.path.join(self.data_path, 'data_r/%s/%s' % ('our', name)))
            self.__crop_img__(defocused_img, coor, name, 'defocused')

            for cp_item in self.cp_items:
                print(cp_item)
                img = cv2.imread(os.path.join(self.data_path, 'log/%s/%s/%s' % (cp_item, item, name)))
                self.__crop_img__(img, coor, name, cp_item.replace('/', '_'))

            if item is '3D_to_3D':
                defocused_img = cv2.imread(os.path.join(self.data_path, 'data_r/3D_label/%s' %name))
                self.__crop_img__(defocused_img, coor, name, 'focused')


if __name__ == '__main__':
    cropImg = CropImg()
    cropImg.crop_items('our_to_3D')