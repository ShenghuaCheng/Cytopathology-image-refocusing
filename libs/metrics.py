import numpy as np
import cv2
import skimage
import skimage.metrics
import math
from sklearn import metrics

# NR image evaluate , sd and vo
class Metrics():

    """
        各种评价图像的指标
    """
    def __init__(self):

        pass

    def brenner(self, img, multi_channel=False):
        '''
        :param img:
        :param multi_channel:
        :return:
        '''
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = np.shape(img)
        out = 0
        for x in range(0, shape[0] - 2):
            for y in range(0, shape[1]):
                out += (int(img[x + 2, y]) - int(img[x, y])) ** 2
        return out

    def Laplacian(self, img, multi_channel=False):
        '''
        :param img:
        :param multi_channel:
        :return:
        '''
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(img, cv2.CV_64F).var()

    def SMD(self, img, multi_channel=False):
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = np.shape(img)
        out = 0
        for x in range(1, shape[0] - 1):
            for y in range(0, shape[1]):
                out += math.fabs(int(img[x, y]) - int(img[x, y - 1]))
                out += math.fabs(int(img[x, y] - int(img[x + 1, y])))
        return out

    def SMD2(self, img, multi_channel=False):
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = np.shape(img)
        out = 0
        for x in range(0, shape[0] - 1):
            for y in range(0, shape[1] - 1):
                out += math.fabs(int(img[x, y]) - int(img[x + 1, y])) * math.fabs(int(img[x, y] - int(img[x, y + 1])))
        return out

    def energy(self, img, multi_channel=False):
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = np.shape(img)
        out = 0
        for x in range(0, shape[0] - 1):
            for y in range(0, shape[1] - 1):
                out += ((int(img[x + 1, y]) - int(img[x, y])) ** 2) * ((int(img[x, y + 1] - int(img[x, y]))) ** 2)
        return out

    def vollath(self, img, img1=None, multi_channel=False):
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s = img.shape
        u = np.mean(img)
        img1 = np.zeros(s, )
        img1[0: s[0] - 1, :] = img[1: s[0], :]
        out = np.sum(np.multiply(img, img1)) - s[0] * s[1] * (u ** 2)
        return np.sqrt(out / (s[1] * (s[0] - 1))) if out >= 0 else 0

    #spatial frequency 评价图像的梯度分布
    def spatial_frequency(self, img, multi_channel=False):
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.int32(img)
        rf, cf = 0, 0
        k = 0
        for i in range(0, len(img)):
            for j in range(1, len(img[i])):
                rf += math.pow(img[i, j] - img[i, j - 1], 2)
                k += 1
        rf /= k

        k = 0
        for i in range(1, len(img)):
            for j in range(0, len(img[i])):
                cf += math.pow(img[i, j] - img[i - 1, j], 2)
                k += 1
        cf /= k

        return math.sqrt(rf + cf)

    def correlation_coe(self, ref, dist, multi_channel=False):
        if ref is None or dist is None:
            return 0
        if multi_channel:
            ref = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
            dist = cv2.cvtColor(dist, cv2.COLOR_RGB2GRAY)
        s = np.shape(ref)
        a_avg = np.sum(ref) / (s[0] * s[1])
        b_avg = np.sum(dist) / (s[0] * s[1])

        ta = ref - a_avg
        tb = dist - b_avg

        cov_ab = np.sum(ta * tb)

        sq = np.sqrt(np.sum(ta * ta) * np.sum(tb * tb))
        corr_factor = cov_ab / sq
        return corr_factor

    # 计算图像的标准差
    def standard_deviation(self, img, img1=None, multi_channel=False):
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        s = img.shape
        img = img - np.mean(img)
        sd = np.sqrt(np.sum(np.multiply(img, img)) / (s[0] * s[1]))
        return sd


    # 计算图像的平均梯度 (AG)
    def average_gradient(self, img, multi_channel=False):
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.int32(img)
        ag = 0
        for i in range(1, len(img)):
            for j in range(1, len(img[i])):
                dx = img[i, j] - img[i - 1, j]
                dy = img[i, j] - img[i, j - 1]
                ds = np.sqrt((pow(dx, 2) + pow(dy, 2)) / 2)
                ag += ds
        return ag / ((len(img) - 1) * (len(img[0]) - 1))


    #计算两个图像的标准化互信息
    def nmi(self , img_a, img_b, multi_channel=False):
        if img_a is None or img_b is None:
            return 0
        if multi_channel:
            img_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
            img_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)

        img_a = img_a.flatten()
        img_b = img_b.flatten()
        return metrics.normalized_mutual_info_score(img_a , img_b , average_method = 'arithmetic')

    # 计算图像的熵
    def entropy(self, img, img1=None, multi_channel=False):
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        temp = np.zeros((256,), dtype=np.float32)
        k = 0
        for i in range(len(img)):
            for j in range(len(img[i])):
                val = img[i, j]
                temp[val] = temp[img[i, j]] + 1
                k = k + 1

        for i in range(len(temp)):
            temp[i] = temp[i] / k

        en = 0
        for i in range(len(temp)):
            if temp[i] != 0:
                en = en - temp[i] * (math.log(temp[i]) / math.log(2.0))

        return en

    # 求取图像的结构化损失
    def ssim_m(self, ref, dist, multi_channel=True):
        if ref is None or dist is None:
            return 0
        return skimage.metrics.structural_similarity(ref, dist, multichannel=multi_channel)

    def psnr_m(self, ref, dist, multi_channel=True):
        '''
        :param ref:
        :param dist:
        :param multi_channel:
        :return:
        '''
        if ref is None or dist is None:
            return 0
        return skimage.metrics.peak_signal_noise_ratio(ref, dist, )

    #计算图像的视觉信息保真度
    # def vifp_mscale(self , ref, dist , channel = 1):
    #     if channel != 1:
    #         ref = cv2.cvtColor(ref , cv2.COLOR_RGB2GRAY)
    #         dist = cv2.cvtColor(dist , cv2.COLOR_RGB2GRAY)
    #
    #     sigma_nsq = 2
    #     eps = 1e-10
    #
    #     num = 0.0
    #     den = 0.0
    #     for scale in range(1, 5):
    #
    #         N = 2 ** (4 - scale + 1) + 1
    #         sd = N / 5.0
    #
    #         if (scale > 1):
    #             ref = scipy.ndimage.gaussian_filter(ref, sd)  # gaussian_filter
    #             dist = scipy.ndimage.gaussian_filter(dist, sd)
    #             ref = ref[::2, ::2]  # 进行两倍的下采样
    #             dist = dist[::2, ::2]
    #
    #         mu1 = scipy.ndimage.gaussian_filter(ref, sd)  # gaussion_filter
    #         mu2 = scipy.ndimage.gaussian_filter(dist, sd)
    #         mu1_sq = mu1 * mu1
    #         mu2_sq = mu2 * mu2
    #         mu1_mu2 = mu1 * mu2
    #         sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
    #         sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
    #         sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2
    #
    #         sigma1_sq[sigma1_sq < 0] = 0  # 将小于0的项置零
    #         sigma2_sq[sigma2_sq < 0] = 0
    #
    #         g = sigma12 / (sigma1_sq + eps)
    #         sv_sq = sigma2_sq - g * sigma12
    #
    #         g[sigma1_sq < eps] = 0
    #         sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
    #         sigma1_sq[sigma1_sq < eps] = 0
    #
    #         g[sigma2_sq < eps] = 0
    #         sv_sq[sigma2_sq < eps] = 0
    #
    #         sv_sq[g < 0] = sigma2_sq[g < 0]
    #         g[g < 0] = 0
    #         sv_sq[sv_sq <= eps] = eps
    #
    #         num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
    #         den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
    #
    #     vifp = num / den
    #
    #     return vifp

me = Metrics()

metrics_dict = {
    'ssim': me.ssim_m,
    'psnr': me.psnr_m,
    'cc':   me.correlation_coe,
    'nmi':  me.nmi,
    'sd':   me.standard_deviation,
    'vo':   me.vollath,
}

def cal_metrics(img_source, img_target=None, choose_metrics=['ssim', 'psnr', 'sd', 'vo']):
    '''
    :param img_source: BGR img
    :param img_target:
    :param choose_metrics:
    :return: dict
    '''
    assert (img_source is not None) or (img_target is not None)
    img_source_gray = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
    if img_target is not None:
        img_target_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
    else:
        img_target_gray = None

    re_dict = dict()
    for cm in choose_metrics:
        re_dict[cm] = metrics_dict[cm](img_source_gray, img_target_gray)

    return re_dict

if __name__ == '__main__':
    # use example

    img1 = np.zeros((512, 512, 3), dtype=np.uint8)
    img2 = np.zeros((512, 512, 3), dtype=np.uint8)

    from tqdm import tqdm

    for i in tqdm(range(1, 1000)):
        v_psnr = me.psnr_m(img1,  img2)
        v_ssim = me.ssim_m(img1,  img2)