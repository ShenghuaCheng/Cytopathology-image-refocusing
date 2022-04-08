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

    # mark
    mark = '在clear_stage2(att_loss&no_rb)的基础上进行为微调，力争结果结果能够得到提升，提高对抗损失比例'

    style_B = ['3D_to_3D', 'our_to_3D']
    style_B_ratio = [0.5, 0.5]
    style_S = '3D_label'
    data_path = os.path.join(pre, 'data_r')
    log_path = os.path.join(pre, 'log/clear_stage2(3kernel)_for_hot')
    txt_path = os.path.join(pre, 'data_r/txt')

    # generator net choose
    g_net = 14  # 0: build_generator_clear
               # 1: build_generator_clear_lightR
               # 2: build_generator_clear_dense
               # 3: build_generator_clear_complex
               # 4: srgan
               # 5: denseunet

    d_net = 0  # 0: circleGAN discriminator
               # 1: srGAN discriminator

    # pretrain_model set
    pretrained_flag = True
    epoch, batch_i, step = 2, 1, 366000
    wei_pre_path = os.path.join(pre, 'log/clear_stage1(3kernel)/wei_log')
    d_B_weight = os.path.join(wei_pre_path, 'd_B_%d_%d_%d.tf' % (epoch, batch_i, step))
    d_S_weight = os.path.join(wei_pre_path, 'd_S_%d_%d_%d.tf' % (epoch, batch_i, step))
    g_BS_weight = os.path.join(wei_pre_path, 'g_BS_%d_%d_%d.tf' % (epoch, batch_i, step))
    g_SB_weight = os.path.join(wei_pre_path, 'g_SB_%d_%d_%d.tf' % (epoch, batch_i, step))
    combined_weight = os.path.join(wei_pre_path, 'combined_%d_%d_%d.tf' % (epoch, batch_i, step))

    channels = 3

    # attention mask loss
    att_loss = True
    if att_loss:
        lambda_att = 30
        col_thre = 16
        vol_thre = 100
        att_path = os.path.join(pre, 'data_r/defocus_estimate/3D')

    detail = True
    scale_num = 3

    # Input shape
    img_rows = 256
    img_cols = 256
    img_res = (img_rows, img_cols)
    img_shape = (img_rows, img_cols, channels)

    # Calculate output shape of D (PatchGAN)
    patch = int(img_rows / (2 ** 4))
    disc_patch = (patch, patch, 1)

    # Number of filters in the first layer of G and D
    gf = 32
    df = 64
    denseblocks = 2

    dis_lr_step = 4

    g_lr = 0.5 * 1e-4
    d_lr = 0.1 * 1e-4

    R_CHANNEL = 0
    B_CHANNEL = 2

    # Loss weights
    lambda_adv = 1.0  # adv loss
    lambda_cycle = 10.0  # Cycle-consistency loss
    lambda_id = 1.0    # Identity loss
    lambda_l1 = 20  # l1 loss

    optimizer = 'Adam'  # (0.0002, 0.5)


    epochs = 30
    cal_metrics_batch_size = 1000
    cal_metrics_random = False
    batch_size = 2
    sample_interval = 3000
    weight_interval = 3000
    nums_per_epoch = 60000

    gpus = '0'