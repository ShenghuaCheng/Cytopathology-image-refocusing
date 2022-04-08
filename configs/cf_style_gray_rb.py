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
    style_3D = '3D'
    style_our = 'our'
    data_path = os.path.join(pre, 'data_r')
    log_path = os.path.join(pre, 'log/style_rb')
    txt_path = os.path.join(pre, 'data_r/txt')

    # Input shape
    img_rows = 256
    img_cols = 256
    channels = 3
    img_shape = (img_rows, img_cols, channels)

    # Calculate output shape of D (PatchGAN)
    patch = int(img_rows / (2 ** 4))
    disc_patch = (patch, patch, 1)

    # Number of filters in the first layer of G and D
    gf = 32
    df = 64


    g_net = 6  # 0: build_generator_clear
               # 1: build_generator_clear_lightR
               # 2: build_generator_clear_dense
               # 3: build_generator_clear_complex
               # 4: srgan
               # 5: denseunet
               # 6: build generator_style

    d_net = 2  # 0: complex circleGAN discriminator
               # 1: srGAN discriminator
               # 2: original cyclegan dis

    R_CHANNEL = 0
    B_CHANNEL = 2

    # Loss weights
    lambda_adv = 1    # adv loss
    lambda_cycle = 10.0  # Cycle-consistency loss
    lambda_id = 1    # Identity loss
    lambda_rb = 0.1  # red mask loss
    # lambda_rb = 0
    # lambda_gray = 1
    lambda_gray = 0

    optimizer = 'Adam'  # (0.0002, 0.5)
    # g_lr = 4 * 10e-4
    d_lr = 1 * 10e-4

    epochs = 100
    batch_size = 16
    sample_interval = 200
    weight_interval = 2500

    nums_per_epoch = 90000

    gpus = '1'

    pretrained_flag = True
    if pretrained_flag:
        epoch, batch_i, step = 2, 1252, 12500
        wei_pre_path = os.path.join(log_path, 'wei_log')
        d_our_weight = os.path.join(wei_pre_path, 'd_our_%d_%d_%d.h5' % (epoch, batch_i, step))
        d_3D_weight = os.path.join(wei_pre_path, 'd_3D_%d_%d_%d.h5' % (epoch, batch_i, step))
        g_our_3D_weight = os.path.join(wei_pre_path, 'g_our_3D_%d_%d_%d.h5' % (epoch, batch_i, step))
        g_3D_our_weight = os.path.join(wei_pre_path, 'g_3D_our_%d_%d_%d.h5' % (epoch, batch_i, step))
        combined_weight = os.path.join(wei_pre_path, 'combined_%d_%d_%d.h5' % (epoch, batch_i, step))