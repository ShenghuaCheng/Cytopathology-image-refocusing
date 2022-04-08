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

    mask = 'no mask'
    style_B = '3D_to_3D'
    style_S = '3D_label'
    data_path = os.path.join(pre, 'data_r')
    log_path = os.path.join(pre, 'log/clear_stage1_nocircle(unet&multi_scale&3kernel)_v1')
    txt_path = os.path.join(pre, 'data_r/txt')

    # Input shape
    img_rows = 256
    img_cols = 256
    img_res = (img_rows, img_cols)
    channels = 3

    # train strategy
    adv_flag = True

    # attention mask loss
    att_loss = False
    if att_loss:
        lambda_att = 1
        col_thre = 16
        vol_thre = 100
        att_path = os.path.join(pre, 'data_r/defocus_estimate/3D')

    # create detail branch
    detail = True
    if detail:
        scale_num = 3

    # generator net choose
    g_net = 14  # 0: build_generator_clear
               # 1: build_generator_clear_lightR
               # 2: build_generator_clear_dense
               # 3: build_generator_clear_complex
               # 4: srgan
               # 5: denseunet

    d_net = 0  # 0: circleGAN discriminator
               # 1: srGAN discriminator


    block_shape = (256, 256, 3)
    img_shape = (img_rows, img_cols, channels)

    # Calculate output shape of D (PatchGAN)
    patch = int(img_rows / (2 ** 4))
    disc_patch = (patch, patch, 1)

    # Number of filters in the first layer of G and D
    gf = 32
    denseblocks = 2
    df = 64

    g_lr = 1 * 1e-4
    d_lr = 0.8 * 1e-4

    R_CHANNEL = 0
    B_CHANNEL = 2

    # Loss weights
    lambda_adv = 0.001  # adv loss
    lambda_l1 = 1.2  # l1 loss

    optimizer = 'Adam'  # (0.0002, 0.5)


    # cal metrics config:
    cal_metrics = ['ssim', 'psnr']
    cal_metrics_batch_size = 2000
    cal_metrics_random = False


    epochs = 100
    batch_size = 6
    sample_interval = 5000
    weight_interval = 5000

    nums_per_epoch = 60000

    gpus = '0'


    pretrained_flag = False
    if pretrained_flag:
        wei_path = os.path.join(log_path, 'wei_log')
        step = 19000
        log_step = '12_1012_19000.tf'
        g_BS_weight = '%s/g_BS_%s' % (wei_path, log_step)
        d_S_weight = '%s/d_S_%s' % (wei_path, log_step)
        combined_weight = '%s/combined_%s' % (wei_path, log_step)