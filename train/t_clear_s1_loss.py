'''
add gray img loss
'''

from configs.cf_clear_stage1_loss import config as conf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpus

from mmodels.mb import d_nets_dict, g_nets_dict
from mmodels.models import ColorLayer, construct_Color_Model
from dls.dl_clear_stage1 import DataLoaderClear, sample_images_clear_loss, val_clear_loss
from train.t_utils import log_write_clear_stage1, val_metrics_stage1_loss

import datetime
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorboardX import SummaryWriter

# constrcut model
g_optimizer = Adam(conf.g_lr, 0.5) if conf.optimizer is 'Adam' else None
d_optimizer = Adam(conf.d_lr, 0.5) if conf.optimizer is 'Adam' else None
# Build and compile the discriminators
d_B = d_nets_dict[conf.d_net](conf)
d_S = d_nets_dict[conf.d_net](conf)
d_B.compile(loss='mse', optimizer=d_optimizer, metrics=['accuracy'])
d_S.compile(loss='mse', optimizer=d_optimizer, metrics=['accuracy'])

# -------------------------
# Construct Computational
#   Graph of Generators
# -------------------------
r_layer = ColorLayer(conf.R_CHANNEL)
r_model = construct_Color_Model(r_layer)
b_layer = ColorLayer(conf.B_CHANNEL)
b_model = construct_Color_Model(b_layer)
# Build the generators
g_BS = g_nets_dict[conf.g_net](conf)
g_SB = g_nets_dict[conf.g_net](conf)

# Input images from both domains
if conf.att_flag:
    img_B = Input(shape=conf.img_shape[: 2] + (4, ))
    img_S = Input(shape=conf.img_shape[: 2] + (4, ))
else:
    img_B = Input(shape=conf.img_shape)
    img_S = Input(shape=conf.img_shape)

# Translate images to the other domain
fake_S = g_BS(img_B[:, :, :, : 3])
fake_B = g_SB(img_S[:, :, :, : 3])

fake_S_r = r_layer(fake_S)
fake_B_r = r_layer(fake_B)

fake_S_b = b_layer(fake_S)
fake_B_b = b_layer(fake_B)

# Translate images back to original domain
reconstr_B = g_SB(fake_S)
reconstr_S = g_BS(fake_B)
# Identity mapping of images
img_B_id = g_SB(img_B[:, :, :, : 3])
img_S_id = g_BS(img_S[:, :, :, : 3])

# For the combined model we will only train the generators
d_B.trainable = False
d_S.trainable = False

# Discriminators determines validity of translated images
valid_B = d_B(fake_B)
valid_S = d_S(fake_S)

att_B = tf.stack([img_B[:, :, :, 3],
                  img_B[:, :, :, 3],
                  img_B[:, :, :, 3]], axis=-1)
att_S = tf.stack([img_S[:, :, :, 3],
                  img_S[:, :, :, 3],
                  img_S[:, :, :, 3]], axis=-1)
att_fake_S = ((fake_S * 0.5 + 0.5) * (att_B * 0.5 + 0.5) - 0.5) * 2
att_fake_B = ((fake_B * 0.5 + 0.5) * (att_S * 0.5 + 0.5) - 0.5) * 2

# Combined model trains generators to fool discriminators
combined = Model(inputs=[img_B, img_S],
                 outputs=[valid_B, valid_S,
                          reconstr_B, reconstr_S,
                          img_B_id, img_S_id,
                          fake_S_r, fake_B_r,
                          fake_S_b, fake_B_b,
                          fake_S, fake_B,
                          att_fake_S, att_fake_B])
combined.compile(loss=['mse', 'mse',
                       'mae', 'mae',
                       'mae', 'mae',
                       'mse', 'mse',
                       'mse', 'mse',
                       'mae', 'mae',
                       'mae', 'mae'],
                 loss_weights=[conf.lambda_adv, conf.lambda_adv,
                               conf.lambda_cycle, conf.lambda_cycle,
                               conf.lambda_id, conf.lambda_id,
                               conf.lambda_red, conf.lambda_red,
                               conf.lambda_red, conf.lambda_red,
                               conf.lambda_l1, conf.lambda_l1,
                               conf.lambda_att, conf.lambda_att], optimizer=g_optimizer)

if conf.pretrained_flag:
    d_B.load_weights(conf.d_B_weight)
    d_S.load_weights(conf.d_S_weight)
    g_BS.load_weights(conf.g_BS_weight)
    g_SB.load_weights(conf.g_SB_weight)
    combined.load_weights(conf.combined_weight)


models = {'g_BS': g_BS, 'g_SB': g_SB, 'combined': combined, 'd_B': d_B, 'd_S': d_S, 'r_model': r_model, 'b_model': b_model}

img_log = '%s/img_log' % (conf.log_path, )
wei_log = '%s/wei_log' % (conf.log_path, )
run_log = '%s/run_log' % (conf.log_path, )
if not os.path.exists(img_log):
    os.makedirs(img_log)
if not os.path.exists(wei_log):
    os.makedirs(wei_log)

metrics_txt_path = os.path.join(conf.log_path, 'metrics_log.txt')
# write config
with open(os.path.join(conf.log_path, 'conf.txt'), 'w') as conf_file:
    for c in conf.__dict__:
        if c != '__dict__':
            conf_file.write('%s:%s\n' % (c, getattr(conf, c)))

log_write = SummaryWriter(run_log)

step = 0
if conf.pretrained_flag:
    step = conf.step

if __name__ == '__main__':

    start_time = datetime.datetime.now()

    # Adversarial loss ground truths
    disc_patch = conf.disc_patch
    if conf.upscale:
        disc_patch = (int(disc_patch[0] * conf.upscale_ratio),
                      int(disc_patch[0] * conf.upscale_ratio),
                      1)
    valid = np.ones((conf.batch_size,) + disc_patch)
    fake = np.zeros((conf.batch_size,) + disc_patch)
    g_loss = np.zeros((12, ))
    d_loss = np.zeros((12, ))
    for epoch in range(conf.epochs):
        if conf.pretrained_flag and epoch < conf.epoch:
            continue
        # construct dataloader
        dataloader = DataLoaderClear(conf)
        imgs_item = (None, None, None, None)
        if conf.att_flag:
            imgs_item = imgs_item + (None, None)

        for batch_i, imgs_item in enumerate(dataloader.load_batch_train(conf.batch_size, is_testing=False)):
            if conf.pretrained_flag and batch_i < conf.batch_i:
                continue
            imgs_B, imgs_S, imgs_BL, imgs_SL = imgs_item[0], imgs_item[1], imgs_item[2], imgs_item[3]
            B_g_BS_in = imgs_B
            S_g_SB_in = imgs_S

            if conf.att_flag:
                imgs_B_att, imgs_S_att = imgs_item[4], imgs_item[5]

                B_g_BS_in = np.concatenate([imgs_B, imgs_B_att], axis=-1)
                S_g_SB_in = np.concatenate([imgs_S, imgs_S_att], axis=-1)

                att_B = np.concatenate([imgs_B_att, imgs_B_att, imgs_B_att], axis=-1)
                att_S = np.concatenate([imgs_S_att, imgs_S_att, imgs_S_att], axis=-1)
                att_BL = ((imgs_BL * 0.5 + 0.5) * (att_B * 0.5 + 0.5) - 0.5) * 2
                att_SL = ((imgs_SL * 0.5 + 0.5) * (att_S * 0.5 + 0.5) - 0.5) * 2

            # ----------------------
            #  Train Discriminators
            # ----------------------

            # Translate images to opposite domain
            fake_S = g_BS.predict(imgs_B)
            fake_B = g_SB.predict(imgs_S)

            # generate rb mask
            imgs_B_r = r_model(imgs_B)
            imgs_B_b = b_model(imgs_B)

            imgs_S_r = r_model(imgs_S)
            imgs_S_b = b_model(imgs_S)

            # Train the discriminators (original images = real / translated = Fake)
            dB_loss_real = d_B.train_on_batch(imgs_B, valid)
            dB_loss_fake = d_B.train_on_batch(fake_B, fake)
            dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

            dS_loss_real = d_S.train_on_batch(imgs_S, valid)
            dS_loss_fake = d_S.train_on_batch(fake_S, fake)
            dS_loss = 0.5 * np.add(dS_loss_real, dS_loss_fake)

            # Total disciminator loss
            d_loss = 0.5 * np.add(dB_loss, dS_loss)

            # ------------------
            #  Train Generators
            # ------------------
            # results = combined.predict([B_g_BS_in, S_g_SB_in])
            g_loss = combined.train_on_batch([B_g_BS_in, S_g_SB_in],
                                              [valid, valid,  # valid_A , valid_B
                                               imgs_B, imgs_S,  # reconstr_A , reconstr_B
                                               imgs_B, imgs_S,  # img_A_id , img_B_id
                                               imgs_B_r, imgs_S_r,
                                               imgs_B_b, imgs_S_b,  # fake_B_Red , fake_A_Red
                                               imgs_BL, imgs_SL,
                                               att_BL, att_SL])  # fake_B , fake_A

            elapsed_time = datetime.datetime.now() - start_time

            # Plot the progress
            print \
                ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc:%3d%%] "
                 "[G loss: %05f, adv: %05f, recon: %05f, id: %05f"
                 ", red:%05f, blu:%05f, l_l1:%05f, att:%05f] time: %s " \
                % (epoch, conf.epochs,
                   batch_i, dataloader.n_batches,
                   d_loss[0], 100 * d_loss[1],
                   g_loss[0],
                   np.mean(g_loss[1:3]),
                   np.mean(g_loss[3:5]),
                   np.mean(g_loss[5:7]),
                   np.mean(g_loss[7:9]),
                   np.mean(g_loss[9: 11]),
                   np.mean(g_loss[11: 13]),
                   np.mean(g_loss[13: 15]),
                   elapsed_time))

            # If at save interval => save generated image samples
            if step % conf.sample_interval == 0 and step != 0:
                sample_images_clear_loss(epoch, batch_i, step, dataloader, combined, r_model, b_model, img_log, conf)
                vd_loss, vg_loss = val_clear_loss(dataloader, models, valid, fake, conf)
                metrics = val_metrics_stage1_loss(dataloader, g_BS, cal_batch=conf.cal_metrics_batch_size,
                                             conf=conf, random=conf.cal_metrics_random)

                # write metrics
                metrics_txt = open(metrics_txt_path, 'a+')
                metrics_txt.write('%.4f,%.4f,%.4f,%.4f\n' % (metrics['ssim'], metrics['psnr'], metrics['sd'], metrics['vo']))
                metrics_txt.close()

                log_write_clear_stage1(log_write, g_loss, d_loss, vg_loss, vd_loss, metrics, step)

            if step % conf.weight_interval == 0 and step != 0:
                g_BS.save_weights('%s/g_BS_%d_%d_%d.h5' % (wei_log, epoch, batch_i, step))
                g_SB.save_weights('%s/g_SB_%d_%d_%d.h5' % (wei_log, epoch, batch_i, step))
                d_B.save_weights('%s/d_B_%d_%d_%d.h5' % (wei_log, epoch, batch_i, step))
                d_S.save_weights('%s/d_S_%d_%d_%d.h5' % (wei_log, epoch, batch_i, step))
                combined.save_weights('%s/combined_%d_%d_%d.h5' % (wei_log, epoch, batch_i, step))
            step += 1
