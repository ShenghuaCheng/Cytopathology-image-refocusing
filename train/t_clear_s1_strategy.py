

from configs.cf_clear_stage1 import config as conf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpus

from mmodels.mb import d_nets_dict, g_nets_dict
from dls.dl_clear_stage1 import DataLoaderClear
from train.t_utils import cs1_val_metrics, cs1_sample_img, cs1_val_loss

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

block_B_img_input = Input(shape=(conf.scale_num, ) + conf.img_shape[: 2] + (3, ))
block_S_img_input = Input(shape=(conf.scale_num, ) + conf.img_shape[: 2] + (3, ))
block_B_rect_input = Input(shape=(conf.scale_num, 5))
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


combined_inputs = [img_B, img_S]
combined_outputs = [valid_B, valid_S,
                    reconstr_B, reconstr_S,
                    img_B_id, img_S_id,
                    fake_S, fake_B]


losses = ['mse', 'mse',
          'mae', 'mae',
          'mae', 'mae',
          'mae', 'mae']
losswe = [conf.lambda_adv, conf.lambda_adv,
          conf.lambda_cycle, conf.lambda_cycle,
          conf.lambda_id, conf.lambda_id,
          conf.lambda_l1, conf.lambda_l1]

if conf.detail:
    combined_inputs += [block_B_img_input, block_B_rect_input, block_S_img_input, block_S_rect_input]

if conf.att_loss:
    combined_outputs += [att_fake_S, att_fake_B]
    losses += ['mae', 'mae']
    losswe += [conf.lambda_att, conf.lambda_att]

# Combined model trains generators to fool discriminators
combined_model = Model(inputs=combined_inputs, outputs=combined_outputs)
combined_model.compile(loss=losses, loss_weights=losswe, optimizer=g_optimizer)

if conf.pretrained_flag:
    d_B.load_weights(conf.d_B_weight)
    d_S.load_weights(conf.d_S_weight)
    g_BS.load_weights(conf.g_BS_weight)
    g_SB.load_weights(conf.g_SB_weight)
    combined_model.load_weights(conf.combined_weight)


models = {'g_BS': g_BS, 'g_SB': g_SB, 'combined': combined_model, 'd_B': d_B, 'd_S': d_S}

img_log = '%s/img_log' % (conf.log_path, )
wei_log = '%s/wei_log' % (conf.log_path, )
run_log = '%s/run_log1' % (conf.log_path, )
if not os.path.exists(img_log):
    os.makedirs(img_log)
if not os.path.exists(wei_log):
    os.makedirs(wei_log)

# write config
with open(os.path.join(conf.log_path, 'conf.txt'), 'w') as conf_file:
    for c in conf.__dict__:
        if c != '__dict__':
            conf_file.write('%s:%s\n' % (c, getattr(conf, c)))

log_write = SummaryWriter(run_log)

step = 0
if conf.pretrained_flag:
    step = conf.step + 1

if __name__ == '__main__':

    start_time = datetime.datetime.now()

    # Adversarial loss ground truths
    valid = np.ones((conf.batch_size,) + conf.disc_patch)
    fake = np.zeros((conf.batch_size,) + conf.disc_patch)
    g_loss = np.zeros((12, ))
    d_loss = np.zeros((12, ))
    for epoch in range(conf.epochs):
        # construct dataloader
        dataloader = DataLoaderClear(conf)

        for batch_i, imgs_dict in enumerate(dataloader.load_batch_train(conf.batch_size, is_testing=False)):
            imgs_B = imgs_dict['imgs_B']
            imgs_S = imgs_dict['imgs_S']
            imgs_BL = imgs_dict['imgs_B_label']
            imgs_SL = imgs_dict['imgs_S_label']
            B_g_BS_in = imgs_B
            S_g_SB_in = imgs_S

            if conf.att_loss:
                imgs_B_att = imgs_dict['imgs_B_att']
                imgs_S_att = imgs_dict['imgs_S_att']
                B_g_BS_in = np.concatenate([imgs_B, imgs_B_att], axis=-1)
                S_g_SB_in = np.concatenate([imgs_S, imgs_S_att], axis=-1)
                att_B = np.concatenate([imgs_B_att, imgs_B_att, imgs_B_att], axis=-1)
                att_S = np.concatenate([imgs_S_att, imgs_S_att, imgs_S_att], axis=-1)
                att_BL = ((imgs_BL * 0.5 + 0.5) * (att_B * 0.5 + 0.5) - 0.5) * 2
                att_SL = ((imgs_SL * 0.5 + 0.5) * (att_S * 0.5 + 0.5) - 0.5) * 2

            if conf.detail:
                imgs_B_ker, imgs_B_rect = imgs_dict['imgs_B_det'], imgs_dict['imgs_B_rec']
                imgs_S_ker, imgs_S_rect = imgs_dict['imgs_S_det'], imgs_dict['imgs_S_rec']

            # ----------------------
            #  Train Discriminators
            # ----------------------

            # Translate images to opposite domain
            if conf.detail:
                fake_S = g_BS.predict([imgs_B, imgs_B_ker, imgs_B_rect])
                fake_B = g_SB.predict([imgs_S, imgs_S_ker, imgs_S_rect])
            else:
                fake_S = g_BS.predict(imgs_B)
                fake_B = g_SB.predict(imgs_S)

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
            inputs = [B_g_BS_in, S_g_SB_in]
            outputs = [valid, valid,  # valid_A , valid_B
                       imgs_B, imgs_S,  # reconstr_A , reconstr_B
                       imgs_B, imgs_S,  # img_A_id , img_B_id
                       imgs_BL, imgs_SL]# fake_B , fake_A
            if conf.att_loss:
                outputs += [att_BL, att_SL]
            if conf.detail:
                inputs += [imgs_B_ker, imgs_B_rect, imgs_S_ker, imgs_S_rect]
            g_loss = combined_model.train_on_batch(inputs, outputs)

            elapsed_time = datetime.datetime.now() - start_time

            # Plot the progress
            print \
                ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc:%3d%%] "
                 "[G loss: %05f, adv: %05f, recon: %05f, id: %05f, l_l1:%05f," \
                % (epoch, conf.epochs,
                   batch_i, dataloader.n_batches,
                   d_loss[0], 100 * d_loss[1],
                   g_loss[0],
                   np.mean(g_loss[1:3]),
                   np.mean(g_loss[3:5]),
                   np.mean(g_loss[5:7]),
                   np.mean(g_loss[7:9])), end='')
            if conf.att_loss:
                print(' l_att:%05f' % np.mean(g_loss[9: 11]), end='')
            print('] time: %s' % elapsed_time)

            # If at save interval => save generated image samples
            if step % conf.sample_interval == 0 and step != 0:
                cs1_sample_img(epoch, batch_i, step, dataloader, combined_model, img_log, conf)
                vd_loss, vg_loss = cs1_val_loss(dataloader, models, valid, fake, conf)

                metrics = cs1_val_metrics(dataloader, g_BS, conf=conf)
                # write metrics
                for k in metrics.keys():
                    log_write.add_scalar('metrics/%s' % k, metrics[k], step)

                # write loss
                td_loss, tg_loss = d_loss, g_loss
                log_write.add_scalars('discriminator/loss', {'t_loss': td_loss[0],
                                                             'v_loss': vd_loss[0]}, step)
                log_write.add_scalars('discriminator/acc', {'t_acc': td_loss[1] * 100,
                                                            'v_acc': vd_loss[1] * 100}, step)
                log_write.add_scalars('generator/loss', {'tg_loss': g_loss[0],
                                                         'vg_loss': vg_loss[0]}, step)
                log_write.add_scalars('generator/adv', {'t_adv': np.mean(tg_loss[1:3]),
                                                        'v_adv': np.mean(vg_loss[1: 3])}, step)
                log_write.add_scalars('generator/recon', {'t_recon': np.mean(tg_loss[3: 5]),
                                                          'v_recon': np.mean(vg_loss[3: 5])}, step)
                log_write.add_scalars('generator/id', {'t_id': np.mean(tg_loss[5: 7]),
                                                       'v_id': np.mean(vg_loss[5: 7])}, step)
                log_write.add_scalars('generator/l1', {'t_l1': np.mean(tg_loss[7: 9]),
                                                       'v_l1': np.mean(vg_loss[7: 9])}, step)
                if conf.att_loss:
                    log_write.add_scalars('generator/l1_att', {'t_l1_att': np.mean(tg_loss[9: 11]),
                                                               'v_l1_att': np.mean(vg_loss[9: 11])}, step)

            if step % conf.weight_interval == 0 and step != 0:
                g_BS.save_weights('%s/g_BS_%d_%d_%d.tf' % (wei_log, epoch, batch_i, step))
                g_SB.save_weights('%s/g_SB_%d_%d_%d.tf' % (wei_log, epoch, batch_i, step))
                d_B.save_weights('%s/d_B_%d_%d_%d.tf' % (wei_log, epoch, batch_i, step))
                d_S.save_weights('%s/d_S_%d_%d_%d.tf' % (wei_log, epoch, batch_i, step))
                combined_model.save_weights('%s/combined_%d_%d_%d.tf' % (wei_log, epoch, batch_i, step))
            step += 1
