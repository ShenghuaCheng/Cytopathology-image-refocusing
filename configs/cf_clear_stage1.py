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
    log_path = os.path.join(pre, 'log/clear_stage1(3kernel)')
    txt_path = os.path.join(pre, 'data_r/txt')

    # Input shape
    img_rows = 256
    img_cols = 256
    img_res = (img_rows, img_cols)
    channels = 3

    # train strategy

    # attention mask loss
    att_loss = True
    if att_loss:
        lambda_att = 30
        col_thre = 16
        vol_thre = 100
        att_path = os.path.join(pre, 'data_r/defocus_estimate/3D')

    # create detail branch
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

    pretrained_flag = True
    # 4_6004_246000
    epoch, batch_i, step = 4, 6004, 246000
    wei_pre_path = os.path.join(log_path, 'wei_log')
    d_B_weight = os.path.join(wei_pre_path, 'd_B_%d_%d_%d.tf' % (epoch, batch_i, step))
    d_S_weight = os.path.join(wei_pre_path, 'd_S_%d_%d_%d.tf' % (epoch, batch_i, step))
    g_BS_weight = os.path.join(wei_pre_path, 'g_BS_%d_%d_%d.tf' % (epoch, batch_i, step))
    g_SB_weight = os.path.join(wei_pre_path, 'g_SB_%d_%d_%d.tf' % (epoch, batch_i, step))
    combined_weight = os.path.join(wei_pre_path, 'combined_%d_%d_%d.tf' % (epoch, batch_i, step))

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
    d_lr = 1 * 1e-4

    R_CHANNEL = 0
    B_CHANNEL = 2

    # Loss weights
    lambda_adv = 1.0  # adv loss
    lambda_cycle = 10.0  # Cycle-consistency loss
    lambda_id = 1.0    # Identity loss
    lambda_l1 = 20  # l1 loss

    optimizer = 'Adam'  # (0.0002, 0.5)


    # cal metrics config:
    cal_metrics = ['ssim', 'psnr']
    cal_metrics_batch_size = 2000
    cal_metrics_random = False


    epochs = 100
    batch_size = 1
    sample_interval = 6000
    weight_interval = 6000

    nums_per_epoch = 60000

    gpus = '0'

    # tmp_path = pre + 'paper_stage2/log/tmp'