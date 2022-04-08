'''
test img clear model
'''
import os
from libs.metrics import cal_metrics
from test.cfs_test.cf_t_clear import config as conf

import datetime
import cv2
from tqdm import tqdm
import numpy as np


class TestCpMethod:

    def __init__(self, cp_result_path=None, txt_path=None, label_path=None):

        self.cp_result_path = cp_result_path
        self.txt_path = txt_path
        self.label_path = label_path
        self.name_list = self.__name_list__(self.txt_path)

    def __name_list__(self, txt_path):
        name_list = list()
        tmp = open(txt_path, 'r')
        for line in tmp:
            line = line.strip()
            name_list.append(line)
        tmp.close()
        return name_list

    def test(self):
        metrics_dict = {}
        for me in conf.metrics:
            metrics_dict[me] = []

        start_time = datetime.datetime.now()
        for name in tqdm(self.name_list):
            img = cv2.imread(os.path.join(self.cp_result_path, name))
            label = cv2.imread(os.path.join(self.label_path, name))
            re_dict = cal_metrics(img, label, choose_metrics=conf.metrics)
            for me in conf.metrics:
                metrics_dict[me].append(re_dict[me])
        elapsed_time = datetime.datetime.now() - start_time

        for me in conf.metrics:
            print('%s: %.4f, ' % (me, np.mean(metrics_dict[me])), end='')
        print('time consumption: %s' % elapsed_time)



if __name__ == '__main__':
    item = '3D_to_3D'
    cp_result_path = 'D:/paper_stage2/data_r/3D_to_3D'
    txt_path = 'D:/paper_stage2/data_r/txt/%s_test.txt' % item
    label_path = 'D:/paper_stage2/data_r/3D_label'

    tcm = TestCpMethod(cp_result_path=cp_result_path,
                       txt_path=txt_path,
                       label_path=label_path)
    tcm.test()



