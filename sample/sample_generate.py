import openslide
import os
import numpy as np
import cv2

from libs.datautils import get_binimg

class SampleGenerate:

    def __init__(self, sc=None):
        self.sc = sc

    @staticmethod
    def __read_img__(ors, p, bs):
        img = ors.read_region(p, 0, (bs, bs))
        img = np.array(img)
        img = img[:, :, 0: 3]
        img = img[:, :, ::-1]
        img = cv2.resize(img, (sc.t_bs, sc.t_bs), interpolation=cv2.INTER_CUBIC)
        return img

    def __save_img__(self, ors, p, bs, col_thre, sn, wsi_name, save_path, ors_label=None):
        img = self.__read_img__(ors, p, bs)
        binimg = get_binimg(img, col_thre, sc.vol_thre)
        if np.sum(binimg) > sc.area_thre * pow(sc.t_bs, 2):

            name = '%s_%d_%d.png' % (wsi_name, p[0], p[1])
            cv2.imwrite('%s/%s' % (save_path, name), img)

            if sn is '3D' and ors_label is not None:
                img = self.__read_img__(ors_label, p, bs)
                cv2.imwrite('%s/%s' % (save_path.replace('3D', '3D_label'), name), img)

    def sample_generate(self):

        for i in range(1, 2):

            bs = int((self.sc.t_bs * self.sc.t_res) / self.sc.wsi_conf[i]['res'])
            r_bs = int((1 - self.sc.redun_ratio) * bs)

            save_path = '%s/%s' % (self.sc.sample_save_path, self.sc.wsi_conf[i]['sn'])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                if self.sc.wsi_conf[i]['sn'] is '3D':
                    os.makedirs(save_path.replace('3D', '3D_label'))

            # wsis = self.sc.wsi_conf[i]['train_wsis'] + self.sc.wsi_conf[i]['test_wsis']
            wsis = self.sc.wsi_conf[i]['test_wsis']
            for j in range(len(wsis)):
                if self.sc.wsi_conf[i]['sn'] is '3D':
                    wsi_path = '%s/%s/%s_Wholeslide_0%s' % (
                    self.sc.wsi_conf[i]['wsi_path'], wsis[j], wsis[j], self.sc.wsi_conf[i]['postfix'])
                else:
                    wsi_path = '%s/%s%s' % (self.sc.wsi_conf[i]['wsi_path'], wsis[j], self.sc.wsi_conf[i]['postfix'])

                print(wsi_path)
                ors = openslide.OpenSlide(wsi_path)

                ors_label = None
                if self.sc.wsi_conf[i]['sn'] is '3D':
                    ors_label = openslide.OpenSlide(wsi_path.replace('Wholeslide_0', 'Wholeslide_Extended'))

                size = ors.level_dimensions[0]

                w_nums = int((size[0] - bs) / r_bs + 1)
                h_nums = int((size[1] - bs) / r_bs + 1)

                for ws in range(w_nums):
                    for hs in range(h_nums):
                        print('%d/%d, %d/%d' % (ws, w_nums, hs, h_nums))
                        p = (ws * bs, hs * bs)
                        self.__save_img__(ors, p, bs, self.sc.wsi_conf[i]['col_thre'],
                                     self.sc.wsi_conf[i]['sn'], wsis[j], save_path, ors_label)

if __name__ == '__main__':
    from sample.sample_conf import SampleConfig as sc
    sg = SampleGenerate(sc)
    sg.sample_generate()

