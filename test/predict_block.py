import numpy as np
import cv2
import os

from mmodels.mb import g_nets_dict

class conf:
    gpus = '0'
    mpp = 0.293
    bs = 256
    redun_ratio = 1 / 4.
    redun_s = int(bs * redun_ratio)
    nredu_s = bs - redun_s

def split_block(block):

    block_shape = np.shape(block)

    def wh_num(x):
        return int((x - conf.bs) / conf.nredu_s) + 1

    w_num = wh_num(block_shape[1])
    h_num = wh_num(block_shape[0])


    imgs = list()
    for w_ind in range(w_num):
        for h_ind in range(h_num):
            img = block[h_ind * conf.nredu_s: h_ind * conf.nredu_s + conf.bs,
                        w_ind * conf.nredu_s: w_ind * conf.nredu_s + conf.bs, :]
            imgs.append(img)

    return imgs, w_num, h_num

def merge_block(blocks, w_num, h_num):

    block = np.zeros(((h_num - 1) * conf.nredu_s + conf.bs,
                      (w_num - 1) * conf.nredu_s + conf.bs, 3), dtype=np.uint8)
    for w_ind in range(w_num):
        for h_ind in range(h_num):
            block[h_ind * conf.nredu_s: h_ind * conf.nredu_s + conf.bs,
                  w_ind * conf.nredu_s: w_ind * conf.nredu_s + conf.bs, :] = blocks[w_ind * h_num + h_ind]

    return block

class PredictBlock:

    def __init__(self):
        pass

    def init_model(self, g_net=10, conf=None, pretrained_weight=None):

        model = g_nets_dict[g_net](conf)
        model.load_weights(pretrained_weight)
        return model

    def init_rfn_model(self):
        g_net = 10
        from configs.cf_clear_stage1_nocircle import config as conf
        pretrained_weight = r'D:\paper_stage2\log\clear_stage1_nocircle(no_kernel)_v1\wei_log\g_BS_13_2013_132000.tf'
        self.rfn_model = self.init_model(g_net, conf, pretrained_weight)

    def init_dnn_model(self):
        g_net = 6
        from configs.cf_style_gray_rb import config as conf
        pretrained_weight = r'D:\paper_stage2\log\style_gray_rb\wei_log/g_our_3D_22_1122_56100.h5'
        self.dnn_model = self.init_model(g_net, conf, pretrained_weight)

    def rfn_predict(self, imgs): return self.rfn_model.predict(imgs)
    def dnn_predict(self, imgs): return self.dnn_model.predict(imgs)

    def pre_op(self, imgs): return np.float32(imgs) / 127.5 - 1
    def sub_op(self, imgs): return np.uint8((imgs + 1) * 127.5)


def predict_dnn_block():
    block_path = r'E:\tmp\our'
    saveb_path = r'E:\tmp\our_to_3D'
    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpus

    pb = PredictBlock()
    pb.init_dnn_model()

    names = os.listdir(block_path)
    for name in names:
        block = cv2.imread('%s/%s' % (block_path, name))
        block = cv2.cvtColor(block, cv2.COLOR_BGR2RGB)
        imgs, w_nums, h_nums = split_block(block)
        imgs = pb.pre_op(imgs)

        results = pb.dnn_predict(imgs)
        results = pb.sub_op(results)
        result = merge_block(results, w_nums, h_nums)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        cv2.imwrite('%s/%s' % (saveb_path, name), result)

def predict_rfn_block():
    block_path = r'E:\tmp\our_to_3D'
    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpus

    pb = PredictBlock()
    pb.init_rfn_model()

    name = '1000_24320.jpg'
    block = cv2.imread('%s/%s' % (block_path, name))
    block = cv2.cvtColor(block, cv2.COLOR_BGR2RGB)
    imgs, w_nums, h_nums = split_block(block)
    imgs = pb.pre_op(imgs)

    results = pb.rfn_predict(imgs)
    results = pb.sub_op(results)
    result = merge_block(results, w_nums, h_nums)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    cv2.imwrite('%s' % name, result)


if __name__ == '__main__':
    predict_rfn_block()
    pass

