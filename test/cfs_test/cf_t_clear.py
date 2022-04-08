import platform
sys = platform.system()
import os
if sys == "Windows":
    pre = 'D:/paper_stage2'
elif sys == "Linux":
    pre = '/mnt/disk_8t/gxb/'
else:
    pre = None
assert pre is not None

class config:

    # path setting
    test_group = 'clear_stage1_nocircle(3kernel)_v1'

    # test item
    item = '3D_to_3D'
    # test img save path
    save_path = os.path.join(pre, 'log/%s/test/%s' % (test_group, item))
    weig_path = os.path.join(pre, 'log/%s/wei_log' % test_group)

    log_path = os.path.join(pre, 'log/%s/test_log' % test_group)

    test_models = ['g_BS_9_7008_189000.tf']
    # data path
    img_path = os.path.join(pre, 'data_r')

    txt_path = os.path.join(pre, 'data_r/txt/%s_test.txt' % item)
    metric_txt_path = os.path.join(pre, 'log/%s/metrics_1000.txt' % test_group)
    img_save_flag = True
    # metrics = ['ssim', 'psnr']  # test metrics
    metrics = ['sd', 'vo']

    detail = True  # 是否增加核质特征信息
    scale_num = 3

    # generator net choose
    g_net = 14 # 0: build_generator_clear
               # 1: build_generator_clear_lightR
               # 2: build_generator_clear_dense
               # 3: build_generator_clear_complex
               # 4: srgan
               # 5: denseunet

    d_net = 0  # 0: circleGAN discriminator
               # 1: srGAN discriminator


    # Input shape
    img_rows = 256
    img_cols = 256
    channels = 3
    img_res = (img_rows, img_cols)
    img_shape = (img_rows, img_cols, channels)

    # Calculate output shape of D (PatchGAN)
    patch = int(img_rows / (2 ** 4))
    disc_patch = (patch, patch, 1)

    # Number of filters in the first layer of G and D
    gf = 32
    denseblocks = 2
    df = 64

    test_nums = 1000
    gpus = '0'