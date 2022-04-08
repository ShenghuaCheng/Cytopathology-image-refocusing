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

    mask = 'add attention mask'
    style_B = '3D_to_3D'
    style_S = '3D_label'
    data_path = os.path.join(pre, 'data_r')
    data_slowly_load = False
    log_path = os.path.join(pre, 'min_log/clear_stage1(att_loss)_v4')
    txt_path = os.path.join(pre, 'data_r/txt')

    # attention mask path
    col_thre = 16
    vol_thre = 100
    att_flag = True
    att_path = os.path.join(pre, 'data_r/defocus_estimate/3D')

    # generator net choose
    g_net = 2  # 0: build_generator_clear
               # 1: build_generator_clear_lightR
               # 2: build_generator_clear_dense
               # 3: build_generator_clear_complex
               # 4: srgan
               # 5: denseunet

    d_net = 0  # 0: circleGAN discriminator
               # 1: srGAN discriminator

    # training strategy
    upscale = False
    upscale_ratio = 2.0

    pretrained_flag = False
    # combined_1_1_10000
    epoch, batch_i, step = 5, 820, 5100
    wei_pre_path = os.path.join(log_path, 'wei_log')
    d_B_weight = os.path.join(wei_pre_path, 'd_B_%d_%d_%d.h5' % (epoch, batch_i, step))
    d_S_weight = os.path.join(wei_pre_path, 'd_S_%d_%d_%d.h5' % (epoch, batch_i, step))
    g_BS_weight = os.path.join(wei_pre_path, 'g_BS_%d_%d_%d.h5' % (epoch, batch_i, step))
    g_SB_weight = os.path.join(wei_pre_path, 'g_SB_%d_%d_%d.h5' % (epoch, batch_i, step))
    combined_weight = os.path.join(wei_pre_path, 'combined_%d_%d_%d.h5' % (epoch, batch_i, step))

    # Input shape
    img_rows = 256
    img_cols = 256
    img_res = (img_rows, img_cols)
    channels = 3
    img_shape = (img_rows, img_cols, channels)

    # Calculate output shape of D (PatchGAN)
    patch = int(img_rows / (2 ** 4))
    disc_patch = (patch, patch, 1)

    # Number of filters in the first layer of G and D
    gf = 32
    denseblocks = 2
    df = 64

    g_lr = 2 * 1e-4
    d_lr = 1 * 1e-4

    R_CHANNEL = 0
    B_CHANNEL = 2

    # Loss weights
    lambda_adv = 1.0  # adv loss
    lambda_cycle = 10.0  # Cycle-consistency loss
    lambda_id = 1.0    # Identity loss
    lambda_red = 0.1  # red mask loss
    lambda_blu = 0.1  # blue mask
    lambda_l1 = 20  # l1 loss
    lambda_att = 30.0  # att mask loss

    optimizer = 'Adam'  # (0.0002, 0.5)

    epochs = 100
    batch_size = 7
    cal_metrics_batch_size = 100
    cal_metrics_random = False
    sample_interval = 100
    weight_interval = 100

    nums_per_epoch = 60000

    gpus = '0'

    # tmp_path = pre + 'paper_stage2/log/tmp'