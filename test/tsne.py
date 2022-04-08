import cv2
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.manifold import TSNE


class TsnePlt:

    def __init__(self):

        self.data_path = 'D:/paper_stage2/data_r'
        self.domains = ['3D', '3D_to_3D', 'our', 'our_to_3D']
        self.domain_txts = ['3D', '3D', 'our', 'our']
        self.cs = ['r', 'g', 'b', 'k']

        self.titles = ['3DHistech', '3DHistech*', 'WNLO', 'WNLO*']

        self.tnse_nums_per_domain = 2

    def __read_item__(self, path):
        img = cv2.imread(path)
        # img = cv2.resize(img, (64, 64))
        # img = img[:, :, : 1]
        # img_flatten = np.reshape(img, (img.shape[0] * img.shape[1] * img.shape[2]))
        # return img_flatten
        return img

    def __read_imgs__(self, ):

        xs = []
        for ind, domain in enumerate(self.domains):
            names = []
            with open(os.path.join(self.data_path, 'txt/%s.txt' % (self.domain_txts[ind])), 'r') as lines:
                for line in lines:
                    names.append(line.strip())

            random.shuffle(names)
            for name in tqdm(names[: self.tnse_nums_per_domain]):
                # print(domain, name)
                x = self.__read_item__(os.path.join(self.data_path, domain, name))
                xs.append(x)
        print(np.shape(xs))
        self.xs = xs


    def draw_tsne(self):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

        for ind, domain in enumerate(self.domains):
            xs_embed = TSNE(n_components=2).fit_transform(
                self.xs[ind * self.tnse_nums_per_domain: (ind + 1) * self.tnse_nums_per_domain])

            i, j = int(ind / 2), int(ind % 2)
            print(i, j)
            axs[i, j].scatter(xs_embed[:, 0], xs_embed[:, 1], c=self.cs[ind], label=domain)
            axs[i, j].set_title(self.titles[ind])

        plt.tight_layout()
        plt.show()


    def __dis_rgb__(self, img):

        s = 256
        c = 256
        dis_r = np.zeros((c, ), dtype=np.uint8)
        dis_g = np.zeros((c, ), dtype=np.uint8)
        dis_b = np.zeros((c, ), dtype=np.uint8)

        for i in range(0, s):
            for j in range(0, s):
                dis_r[img[i, j, 2]] += 1
                dis_g[img[i, j, 1]] += 1
                dis_b[img[i, j, 0]] += 1

        return dis_r, dis_g, dis_b


    def draw_single(self, ):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(2, 2))
        plt.figure(1)

        x = range(0, 256)

        for ind, domain in enumerate(self.domains):
            imgs = self.xs[ind * self.tnse_nums_per_domain: (ind + 1) * self.tnse_nums_per_domain]
            dis_rs, dis_gs, dis_bs = [], [], []
            for img in imgs:
                dis_r, dis_g, dis_b = self.__dis_rgb__(img)
                dis_rs.append(dis_r)
                dis_gs.append(dis_g)
                dis_bs.append(dis_b)

            t = 1
            dis_r = np.mean(dis_rs, axis=0) / t
            dis_g = np.mean(dis_gs, axis=0) / t
            dis_b = np.mean(dis_bs, axis=0) / t


            # i, j = int(ind / 2), int(ind % 2)
            ax = plt.subplot('22%d' % (ind + 1))
            plt.plot(x, dis_r, label='R')
            plt.plot(x, dis_g, label='G')
            plt.plot(x, dis_b, label='B')
            if ind == 0:
                plt.legend()
            ax.set_title(self.titles[ind])

        plt.show()

if __name__ == '__main__':


    p = TsnePlt()
    p.__read_imgs__()
    p.draw_single()
    # p.draw_tsne()