'''
add gray img loss
'''

from configs.cf_clear_stage1_nocircle import config as conf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpus

from mmodels.mb import d_nets_dict, g_nets_dict
from dls.dl_clear_stage1_nocircle import DataLoaderClear
from train.t_utils import cs1_val_metrics, cs1_sample_img_nocircle, cs1_val_loss_nocircle

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
d_S = d_nets_dict[conf.d_net](conf)
d_S.compile(loss='mse', optimizer=d_optimizer, metrics=['accuracy'])

# -------------------------
# Construct Computational
#   Graph of Generators
# -------------------------

# Build the generators
g_BS = g_nets_dict[conf.g_net](conf)
g_BS.summary()

# Input images from both domains
if conf.att_loss:
    img_B = Input(shape=conf.img_shape[: 2] + (4, ))
else:
    img_B = Input(shape=conf.img_shape)

if conf.detail:
    block_B_img_input = Input(shape=(conf.scale_num, ) + conf.img_shape[: 2] + (3, ))
    block_B_rect_input = Input(shape=(conf.scale_num, 5))
# Translate images to the other domain
if conf.att_loss:
    if conf.detail:
        fake_S = g_BS([img_B[:, :, :, 0: 3], block_B_img_input, block_B_rect_input])
    else:
        fake_S = g_BS(img_B[:, :, :, 0: 3])
else:
    if conf.detail:
        fake_S = g_BS([img_B, block_B_img_input, block_B_rect_input])
    else:
        fake_S = g_BS(img_B)

# For the combined model we will only train the generators
d_S.trainable = False

# Discriminators determines validity of translated images
valid_S = d_S(fake_S)

if conf.att_loss:
    att_B = tf.stack([img_B[:, :, :, 3],
                      img_B[:, :, :, 3],
                      img_B[:, :, :, 3]], axis=-1)
    att_fake_S = ((fake_S * 0.5 + 0.5) * (att_B * 0.5 + 0.5) - 0.5) * 2


combined_inputs = img_B

if conf.adv_flag:
    losses = ['mse', 'mae']
    losswe = [conf.lambda_adv, conf.lambda_l1]
    combined_outputs = [valid_S, fake_S]
else:
    losses = ['mae']
    losswe = [conf.lambda_l1]
    combined_outputs = [fake_S]

if conf.detail:
    combined_inputs = [combined_inputs, block_B_img_input, block_B_rect_input]

if conf.att_loss:
    combined_outputs += [att_fake_S]
    losses += ['mae']
    losswe += [conf.lambda_att]

# Combined model trains generators to fool discriminators
combined_model = Model(inputs=combined_inputs, outputs=combined_outputs)
combined_model.compile(loss=losses, loss_weights=losswe, optimizer=g_optimizer)

step = 0
# 如果有预训练权重则加载
if conf.pretrained_flag:
    d_S.load_weights(conf.d_S_weight)
    g_BS.load_weights(conf.g_BS_weight)
    # combined_model.load_weights(conf.combined_weight)
    step = conf.step + 1

models = {'g_BS': g_BS, 'combined': combined_model, 'd_S': d_S}

img_log = '%s/img_log' % (conf.log_path, )
wei_log = '%s/wei_log' % (conf.log_path, )
run_log = '%s/run_log' % (conf.log_path, )
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

if __name__ == '__main__':

    start_time = datetime.datetime.now()

    # Adversarial loss ground truths
    valid = np.ones((conf.batch_size,) + conf.disc_patch)
    fake = np.zeros((conf.batch_size,) + conf.disc_patch)
    g_loss = np.zeros((12, ))
    dS_loss = np.zeros((12, ))
    for epoch in range(conf.epochs):
        # construct dataloader
        dataloader = DataLoaderClear(conf)

        for batch_i, imgs_dict in enumerate(dataloader.load_batch_train(conf.batch_size, is_testing=False)):
            imgs_B = imgs_dict['imgs_B']
            imgs_BL = imgs_dict['imgs_B_label']
            B_g_BS_in = imgs_B

            if conf.att_loss:
                imgs_B_att = imgs_dict['imgs_B_att']
                B_g_BS_in = np.concatenate([imgs_B, imgs_B_att], axis=-1)
                att_B = np.concatenate([imgs_B_att, imgs_B_att, imgs_B_att], axis=-1)
                att_BL = ((imgs_BL * 0.5 + 0.5) * (att_B * 0.5 + 0.5) - 0.5) * 2

            if conf.detail:
                imgs_B_ker, imgs_B_rect = imgs_dict['imgs_B_det'], imgs_dict['imgs_B_rec']

            # ----------------------
            #  Train Discriminators
            # ----------------------

            # Translate images to opposite domain
            if conf.detail:
                fake_S = g_BS.predict([imgs_B, imgs_B_ker, imgs_B_rect])
            else:
                fake_S = g_BS.predict(imgs_B)

            # Train the discriminators (original images = real / translated = Fake)
            if conf.adv_flag:
                dS_loss_real = d_S.train_on_batch(imgs_BL, valid)
                dS_loss_fake = d_S.train_on_batch(fake_S, fake)
                dS_loss = 0.5 * np.add(dS_loss_real, dS_loss_fake)

            # ------------------
            #  Train Generators
            # ------------------
            inputs = B_g_BS_in
            if conf.adv_flag:
                outputs = [valid, imgs_BL]
            else:
                outputs = [imgs_BL]
            if conf.att_loss:
                outputs += [att_BL]
            if conf.detail:
                inputs = [inputs, imgs_B_ker, imgs_B_rect]
            g_loss = combined_model.train_on_batch(inputs, outputs)
            if not conf.adv_flag:
                g_loss = [g_loss[0], 0, ] + g_loss[1: ]

            elapsed_time = datetime.datetime.now() - start_time

            # Plot the progress
            print \
                ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc:%3d%%] "
                 "[G loss: %05f, adv: %05f, l_l1:%05f," \
                % (epoch, conf.epochs,
                   batch_i, dataloader.n_batches,
                   dS_loss[0], 100 * dS_loss[1],
                   g_loss[0], g_loss[1], g_loss[2]), end='')
            if conf.att_loss:
                print(' l_att:%05f' % g_loss[3], end='')
            print('] time: %s' % elapsed_time)

            # If at save interval => save generated image samples
            if step % conf.sample_interval == 0 and step != 0:
                cs1_sample_img_nocircle(epoch, batch_i, step, dataloader, combined_model, img_log, conf)
                # continue
                vd_loss, vg_loss = cs1_val_loss_nocircle(dataloader, models, valid, fake, conf)

                metrics = cs1_val_metrics(dataloader, g_BS, conf=conf)
                # write metrics
                for k in metrics.keys():
                    log_write.add_scalar('metrics/%s' % k, metrics[k], step)

                # write loss
                td_loss, tg_loss = dS_loss, g_loss
                log_write.add_scalars('discriminator/loss', {'t_loss': td_loss[0],
                                                             'v_loss': vd_loss[0]}, step)
                log_write.add_scalars('discriminator/acc', {'t_acc': td_loss[1] * 100,
                                                            'v_acc': vd_loss[1] * 100}, step)
                log_write.add_scalars('generator/loss', {'tg_loss': g_loss[0],
                                                         'vg_loss': vg_loss[0]}, step)
                log_write.add_scalars('generator/adv', {'t_adv': tg_loss[1],
                                                        'v_adv': vg_loss[1]}, step)
                log_write.add_scalars('generator/l1', {'t_l1': tg_loss[2],
                                                       'v_l1': vg_loss[2]}, step)
                if conf.att_loss:
                    log_write.add_scalars('generator/l1_att', {'t_l1_att': tg_loss[3],
                                                               'v_l1_att': vg_loss[3]}, step)

            if step % conf.weight_interval == 0 and step != 0:
                g_BS.save_weights('%s/g_BS_%d_%d_%d.tf' % (wei_log, epoch, batch_i, step))
                d_S.save_weights('%s/d_S_%d_%d_%d.tf' % (wei_log, epoch, batch_i, step))
                combined_model.save_weights('%s/combined_%d_%d_%d.tf' % (wei_log, epoch, batch_i, step))
            step += 1
