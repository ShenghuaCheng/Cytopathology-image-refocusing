from sample.sample_conf import *

import os

if __name__ == '__main__':

    for i in range(0, 1):

        save_path = '%s/%s' % (sample_save_path , wsi_conf[i]['sn'])
        names = os.listdir(save_path)
        names = [x for x in names]

        print(save_path, len(names))
        if not os.path.exists(txt_save_path):
            os.makedirs(txt_save_path)

        txt_total = open('%s/%s.txt' % (txt_save_path , wsi_conf[i]['sn']) , 'w')
        txt_train = open('%s/%s_train.txt' % (txt_save_path, wsi_conf[i]['sn']), 'w')
        txt_test = open('%s/%s_test.txt' % (txt_save_path, wsi_conf[i]['sn']), 'w')

        for name in names:
            name = name.strip()
            txt_total.write('%s/%s\n' % (save_path , name))

            tmp = name[ : name.find('_')]
            if tmp in wsi_conf[i]['train_wsis']:
                txt_train.write('%s/%s\n' % (save_path , name))
            if tmp in wsi_conf[i]['test_wsis']:
                txt_test.write('%s/%s\n' % (save_path, name))

        txt_total.close()
        txt_train.close()
        txt_test.close()
