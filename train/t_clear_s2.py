'''
train clear model by 3D_to_3D and our_to_3D with l1 loss
'''

from configs.cf_clear_stage2 import config as conf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpus

from mmodels.mb import g_nets_dict, d_nets_dict
from dls.dl_clear_stage2 import DataLoader, sample_images_clear, val_clear
from train.t_utils import val_metrics_stage2, log_write_clear_stage2

import datetime
import numpy as np
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
# Build the generators
g_BS = g_nets_dict[conf.g_net](conf)
g_SB = g_nets_dict[conf.g_net](conf)

# Input images from both domains
if conf.att_loss:
    img_B = Input(shape=conf.img_shape[: 2] + (4, ))
    img_S = Input(shape=conf.img_shape[: 2] + (4, ))
else:
    img_B = Input(shape=conf.img_shape)
    img_S = Input(shape=conf.img_shape)


if conf.detail:
    block_B_img_input = Input(shape=(conf.scale_num, ) + conf.img_shape[: 2] + (3, ))
    block_B_rect_input = Input(shape=(conf.scale_num, 5))
    block_S_img_input = Input(shape=(conf.scale_num, ) + conf.img_shape[: 2] + (3, ))
    block_S_rect_input = Input(shape=(conf.scale_num, 5))

# Translate images to the other domain
if conf.att_loss:
    if conf.detail:
        fake_S = g_BS([img_B[:, :, :, 0: 3], block_B_img_input, block_B_rect_input])
        fake_B = g_SB([img_S[:, :, :, 0: 3], block_S_img_input, block_S_rect_input])
    else:
        fake_S = g_BS(img_B[:, :, :, 0: 3])
        fake_B = g_SB(img_S[:, :, :, 0: 3])
else:
    if conf.detail:
        fake_S = g_BS([img_B, block_B_img_input, block_B_rect_input])
        fake_B = g_SB([img_S, block_S_img_input, block_S_rect_input])
    else:
        fake_S = g_BS(img_B)
        fake_B = g_SB(img_S)

# Translate images back to original domain
if conf.detail:
    reconstr_B = g_SB([fake_S, block_B_img_input, block_B_rect_input])
    reconstr_S = g_BS([fake_B, block_S_img_input, block_S_rect_input])
else:
    reconstr_B = g_SB(fake_S)
    reconstr_S = g_BS(fake_B)

# Identity mapping of images
if conf.att_loss:
    if conf.detail:
        img_B_id = g_SB([img_B[:, :, :, 0: 3], block_B_img_input, block_B_rect_input])
        img_S_id = g_BS([img_S[:, :, :, 0: 3], block_S_img_input, block_S_rect_input])
    else:
        img_B_id = g_SB(img_B[:, :, :, 0: 3])
        img_S_id = g_BS(img_S[:, :, :, 0: 3])
else:
    if conf.detail:
        img_B_id = g_SB([img_B, block_B_img_input, block_B_rect_input])
        img_S_id = g_BS([img_S, block_S_img_input, block_S_rect_input])
    else:
        img_B_id = g_SB(img_B)
        img_S_id = g_BS(img_S)

# For the combined model we will only train the generators
d_B.trainable = False
d_S.trainable = False

# Discriminators determines validity of translated images
valid_B = d_B(fake_B)
valid_S = d_S(fake_S)

if conf.att_loss:
    att_B = tf.stack([img_B[:, :, :, 3],
                      img_B[:, :, :, 3],
                      img_B[:, :, :, 3]], axis=-1)
    att_S = tf.stack([img_S[:, :, :, 3],
                      img_S[:, :, :, 3],
                      img_S[:, :, :, 3]], axis=-1)
    att_fake_S = ((fake_S * 0.5 + 0.5) * (att_B * 0.5 + 0.5) - 0.5) * 2
    att_fake_B = ((fake_B * 0.5 + 0.5) * (att_S * 0.5 + 0.5) - 0.5) * 2

# Combined model trains generators to fool discriminators
inputs = [img_B, img_S]

if conf.detail:
    inputs += [block_B_img_input, block_B_rect_input, block_S_img_input, block_S_rect_input]

outputs = [valid_B, valid_S,
           reconstr_B, reconstr_S,
           img_B_id, img_S_id]
loss_med = ['mse', 'mse',
            'mae', 'mae',
            'mae', 'mae']
loss_weights = [conf.lambda_adv, conf.lambda_adv,
                conf.lambda_cycle, conf.lambda_cycle,
                conf.lambda_id, conf.lambda_id]
combined = Model(inputs=inputs, outputs=outputs)
combined.compile(loss=loss_med, loss_weights=loss_weights, optimizer=g_optimizer)

# train supervised model with l1
outputs_l = outputs + [fake_S, fake_B]
loss_med_l = loss_med + ['mae', 'mae']
loss_weights_l = loss_weights + [conf.lambda_l1, conf.lambda_l1]
# if add att_loss
if conf.att_loss:
    outputs_l += [att_fake_S, att_fake_B]
    loss_med_l += ['mae', 'mae']
    loss_weights_l += [conf.lambda_att, conf.lambda_att]
combined_l = Model(inputs=inputs,  outputs=outputs_l)
combined_l.compile(loss=loss_med_l, loss_weights=loss_weights_l, optimizer=g_optimizer)

if conf.pretrained_flag:
    d_B.load_weights(conf.d_B_weight)
    d_S.load_weights(conf.d_S_weight)
    g_BS.load_weights(conf.g_BS_weight)
    g_SB.load_weights(conf.g_SB_weight)
    combined.load_weights(conf.combined_weight)

models = {'g_BS': g_BS, 'g_SB': g_SB, 'combined': combined, 'combined_l': combined_l, 'd_B': d_B, 'd_S': d_S}

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

if __name__ == '__main__':

    start_time = datetime.datetime.now()

    # Adversarial loss ground truths
    valid = np.ones((conf.batch_size,) + conf.disc_patch)
    fake = np.zeros((conf.batch_size,) + conf.disc_patch)
    g_loss = np.zeros((9, ))
    d_loss = np.zeros((2, ))
    for epoch in range(conf.epochs):
        # construct dataloader
        dataloader = DataLoader(conf)
        for batch_i, imgs_dict in enumerate(dataloader.load_batch_train(conf.batch_size)):
            imgs_B1 = imgs_dict['imgs_B1']
            imgs_B1_label = imgs_dict['imgs_B1_label']
            imgs_B2 = imgs_dict['imgs_B2']
            imgs_S = imgs_dict['imgs_S']
            imgs_S_label = imgs_dict['imgs_S_label']

            if conf.detail:
                imgs_B1_det, imgs_B1_rec = imgs_dict['imgs_B1_det'], imgs_dict['imgs_B1_rec']
                imgs_B2_det, imgs_B2_rec = imgs_dict['imgs_B2_det'], imgs_dict['imgs_B2_rec']
                imgs_S_det, imgs_S_rec = imgs_dict['imgs_S_det'], imgs_dict['imgs_S_rec']
            # ----------------------
            #  Train Discriminators
            # ----------------------

            if conf.att_loss:
                imgs_B1_att = imgs_B1[:, :, :, 3:]
                imgs_S_att = imgs_S[:, :, :, 3:]
                imgs_B1_att = np.concatenate([imgs_B1_att, imgs_B1_att, imgs_B1_att], axis=-1)
                imgs_S_att = np.concatenate([imgs_S_att, imgs_S_att, imgs_S_att], axis=-1)
                att_BL = ((imgs_B1_label * 0.5 + 0.5) * (imgs_B1_att * 0.5 + 0.5) - 0.5) * 2
                att_SL = ((imgs_S_label * 0.5 + 0.5) * (imgs_S_att * 0.5 + 0.5) - 0.5) * 2

            # Translate images to opposite domain
            imgs_B_in = np.concatenate([imgs_B1[:, :, :, : 3], imgs_B2], axis=0)
            imgs_S_in = imgs_S[:, :, :, :3]
            if conf.detail:
                imgs_B_det = np.concatenate([imgs_B1_det, imgs_B2_det], axis=0)
                imgs_B_rec = np.concatenate([imgs_B2_rec, imgs_B2_rec], axis=0)
                imgs_B_in = [imgs_B_in, imgs_B_det, imgs_B_rec]

                imgs_S_in = [imgs_S_in, imgs_S_det, imgs_S_rec]

            fake_S = g_BS.predict(imgs_B_in)
            fake_B = g_SB.predict(imgs_S_in)
            if step % conf.dis_lr_step == 0:
                dB_loss_real = d_B.train_on_batch(imgs_B_in, valid)
                dB_loss_fake = d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                dS_loss_real = d_S.train_on_batch(imgs_S_in, valid)
                dS_loss_fake = d_S.train_on_batch(fake_S, fake)
                dS_loss = 0.5 * np.add(dS_loss_real, dS_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dB_loss, dS_loss)

            # ------------------
            #  Train Generators
            # ------------------

            hbs = int(conf.style_B_ratio[0] * conf.batch_size)

            # train with label
            if conf.att_loss:
                imgs_B1_in = imgs_B1
                imgs_S_in = imgs_S
                imgs_B1 = imgs_B1[:, :, :, : 3]
                imgs_S = imgs_S[:, :, :, : 3]

            combine_l_outputs = [valid[: hbs], valid[: hbs],  # valid_A , valid_B
                                 imgs_B1, imgs_S[: hbs],  # reconstr_A , reconstr_B
                                 imgs_B1, imgs_S[: hbs],  # img_A_id , img_B_id
                                 imgs_B1_label, imgs_S_label[: hbs]]
            if conf.att_loss:
                combine_l_outputs += [att_BL, att_SL[: hbs]]

            combine_l_inputs = [imgs_B1_in, imgs_S_in[: hbs]]
            if conf.detail:
                combine_l_inputs += [imgs_B1_det, imgs_B1_rec, imgs_S_det[: hbs], imgs_S_rec[: hbs]]
            g_loss_l = combined_l.train_on_batch(combine_l_inputs, combine_l_outputs)

            # train without label
            if conf.att_loss:  # no label but add meaninglesss img
                imgs_B2_in = np.concatenate([imgs_B2, imgs_B2[:, :, :, 2:]], axis=-1)

            combine_inputs = [imgs_B2_in, imgs_S_in[hbs:]]
            if conf.detail:
                combine_inputs += [imgs_B2_det, imgs_B2_rec, imgs_S_det[hbs:], imgs_S_rec[hbs:]]

            combine_outputs = [valid[hbs:], valid[hbs:], imgs_B2, imgs_S[hbs:], imgs_B2, imgs_S[hbs:]]
            g_loss = combined.train_on_batch(combine_inputs, combine_outputs)

            # get combine_l and combine mean loss
            g_loss[0: 7] = np.array(g_loss_l[0: 7] + g_loss[0: 7]) / 2
            elapsed_time = datetime.datetime.now() - start_time

            # Plot the progress
            print \
                ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc:%3d%%] "
                 "[G loss: %05f, adv: %05f, recon: %05f, id: %05f, l1:%05f, att_loss:%05f] time: %s " \
                % (epoch, conf.epochs,
                   batch_i, dataloader.n_batches,
                   d_loss[0], 100 * d_loss[1],
                   g_loss[0],
                   np.mean(g_loss[1: 3]),
                   np.mean(g_loss[3: 5]),
                   np.mean(g_loss[5: 7]),
                   np.mean(g_loss_l[7: 9]),
                   np.mean(g_loss_l[9: 11]),
                   elapsed_time))

            # If at save interval => save generated image samples
            if step % conf.sample_interval == 0:
                sample_images_clear(epoch, batch_i, step, dataloader, g_BS, g_SB, img_log, batch_size=2, conf=conf)
                metrics = val_metrics_stage2(dataloader, g_BS, conf=conf)
                vd_loss, vg_loss = val_clear(dataloader, models, valid, fake, conf)

                log_write_clear_stage2(log_write, g_loss, d_loss, vg_loss, vd_loss, metrics, step, conf=conf)


            if step % conf.weight_interval == 0:
                g_BS.save_weights('%s/g_BS_%d_%d_%d.h5' % (wei_log, epoch, batch_i, step))
                g_SB.save_weights('%s/g_SB_%d_%d_%d.h5' % (wei_log, epoch, batch_i, step))
                d_B.save_weights('%s/d_B_%d_%d_%d.h5' % (wei_log, epoch, batch_i, step))
                d_S.save_weights('%s/d_S_%d_%d_%d.h5' % (wei_log, epoch, batch_i, step))
                combined.save_weights('%s/combined_%d_%d_%d.h5' % (wei_log, epoch, batch_i, step))

            step += 1
