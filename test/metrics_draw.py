import matplotlib.pyplot as plt
import os


class MetricsDraw:

    def __init__(self):

        self.path = 'D:/paper_stage2'
        self.logs = [
                    # 'min_log/clear_stage1',
                    'min_log/clear_stage1(no_rb&1)',
                    'min_log/clear_stage1(no_rb&3)',
                    # 'min_log/clear_stage1(att_flag)',
                    # 'log/clear_stage2(att_loss&no_rb)',
                    # 'log/clear_stage2(no_rb)'
                    # 'min_log/clear_stage1(scale&att_loss)',
                    # 'min_log/clear_stage1(scale&att_flag)',
                    # 'min_log/clear_stage1(att_loss)',
                    # 'min_log/clear_stage1(att_loss)_v1',
                    # 'min_log/clear_stage1(att_loss)_v4',
                ]

        self.logs_label = [
                    'refocus',
                    'refocus1',
                    # 'refocus+att_flag',
                    # 'refocus+no_rb+att_loss',
                    # 'refocus+no_rb'
                    # 'refocus+upscale+att_loss',
                    # 'refocus+scale+att_flag',
                    # 'refocus+att_loss'
                ]

    def __read_metrics_txt__(self, txt_path):
        '''
        :param txt_path:
        :return:
        '''
        ssims, psnrs, sds, vos = [], [], [], []
        with open(txt_path,  'r') as txt:
            for line in txt:
                line = line.strip()
                uints = line.split(',')
                ssims.append(float(uints[0]))
                psnrs.append(float(uints[1]))
                sds.append(float(uints[2]))
                vos.append(float(uints[3]))
        return {'ssim': ssims, 'psnr': psnrs, 'sd': sds, 'vo': vos}

    def draw(self, nums=1000, metrics=['ssim', 'psnr', 'sd', 'vo']):

        metrics_dict = {}
        for k in metrics:
            metrics_dict[k] = list()

        for log in self.logs:
            ms = self.__read_metrics_txt__(os.path.join(self.path, log, 'metrics_log.txt'))
            for k in metrics_dict.keys():
                metrics_dict[k].append(ms[k])

        fig_format = int('1%d' % len(metrics_dict.keys()))

        for ind, k in enumerate(metrics_dict.keys()):
            plt.subplot(int('%d%d' % (fig_format, ind + 1)))
            plt.ylabel(k, fontsize=30)
            for log_ind in range(0, len(metrics_dict[k])):
                t_nums = min(nums, len(metrics_dict[k][log_ind][: nums]))
                plt.plot([x for x in range(0, t_nums)], metrics_dict[k][log_ind][: t_nums], label=self.logs_label[log_ind])

        plt.show()

if __name__ == '__main__':
    md = MetricsDraw()
    md.draw(metrics=['ssim', 'psnr'])
