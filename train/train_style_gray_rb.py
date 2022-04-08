'''
add gray img loss
'''

from configs.cf_style_gray_rb import config as conf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpus

from mmodels.mb import d_nets_dict, g_nets_dict
from mmodels.models import ColorLayer, GrayLayer, construct_Color_Model
from dls.dl_style_rb import DataLoader

import datetime
import numpy as np
import os
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorboardX import SummaryWriter

# constrcut model
optimizer = Adam(conf.d_lr, 0.5) if conf.optimizer is 'Adam' else None
# Build and compile the discriminators
d_3D = d_nets_dict[conf.d_net](conf)
d_our = d_nets_dict[conf.d_net](conf)

d_3D.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
d_our.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

# -------------------------
# Construct Computational
#   Graph of Generators
# -------------------------
# Build the generators
g_3D_our = g_nets_dict[conf.g_net](conf)
g_our_3D = g_nets_dict[conf.g_net](conf)

# Input images from both domains
img_3D = Input(shape=conf.img_shape)
img_our = Input(shape=conf.img_shape)

# Translate images to the other domain
fake_our = g_3D_our(img_3D)
fake_3D = g_our_3D(img_our)

# gray layer
gray_layer = GrayLayer()
# red and blue mask
r_layer = ColorLayer(conf.R_CHANNEL)
r_model = construct_Color_Model(r_layer)
b_layer = ColorLayer(conf.B_CHANNEL)
b_model = construct_Color_Model(b_layer)

fake_our_gray = gray_layer(fake_our)
fake_3D_gray = gray_layer(fake_3D)

fake_our_r = r_layer(fake_our)
fake_3D_r = r_layer(fake_3D)
fake_our_b = b_layer(fake_our)
fake_3D_b = b_layer(fake_3D)

# Translate images back to original domain
reconstr_3D = g_our_3D(fake_our)
reconstr_our = g_3D_our(fake_3D)
# Identity mapping of images
img_3D_id = g_our_3D(img_3D)
img_our_id = g_3D_our(img_our)

# For the combined model we will only train the generators
d_3D.trainable = False
d_our.trainable = False

# Discriminators determines validity of translated images
valid_3D = d_3D(fake_3D)
valid_our = d_our(fake_our)

# Combined model trains generators to fool discriminators
combined = Model(inputs=[img_3D, img_our],
                 outputs=[valid_3D, valid_our,
                          reconstr_3D, reconstr_our,
                          img_3D_id, img_our_id,
                          fake_our_gray, fake_3D_gray,
                          fake_our_r, fake_3D_r,
                          fake_our_b, fake_3D_b])
combined.compile(loss=['mse', 'mse',
                       'mae', 'mae',
                       'mae', 'mae',
                       'mse', 'mse',
                       'mse', 'mse',
                       'mse', 'mse'],
                 loss_weights=[conf.lambda_adv, conf.lambda_adv,
                               conf.lambda_cycle, conf.lambda_cycle,
                               conf.lambda_id, conf.lambda_id,
                               conf.lambda_gray, conf.lambda_gray,
                               conf.lambda_rb, conf.lambda_rb,
                               conf.lambda_rb, conf.lambda_rb], optimizer=optimizer)
models = {'g_3D_our': g_3D_our, 'g_our_3D': g_our_3D, 'combined': combined, 'd_3D': d_3D, 'd_our': d_our}

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

step = 0
if conf.pretrained_flag:
    d_our.load_weights(conf.d_our_weight)
    d_3D.load_weights(conf.d_3D_weight)
    g_our_3D.load_weights(conf.g_our_3D_weight)
    g_3D_our.load_weights(conf.g_3D_our_weight)
    combined.load_weights(conf.combined_weight)

    pre_epoch = 2
    pre_batch_i = 1252
    step = 12500

def sample_images_style(epoch, batch_i, step, dataloader):

    imgs_3D, imgs_our = dataloader.load_random_images_pair(batch_size=1)
    r, c = 4, 6

    # model predict
    fake_3D = g_our_3D.predict(imgs_our)
    fake_our = g_3D_our.predict(imgs_3D)

    imgs_3D_recon = g_our_3D.predict(fake_our)
    imgs_our_recon = g_3D_our.predict(fake_3D)

    def get_rb_img(img):
        img_r = r_model(img)
        img_b = b_model(img)
        rb_img = np.concatenate([img_r, img_r * 0, img_b], axis=-1)
        return rb_img

    imgs = [imgs_3D, fake_our, imgs_3D_recon,
            imgs_our, fake_3D, imgs_our_recon]

    imgs_id = list()
    for k, img in enumerate(imgs):
        if k % 2 == 0:
            img_id = g_our_3D.predict(img)
        else:
            img_id = g_3D_our.predict(img)
        imgs_id.append(img_id)

    # gray and rb
    gray_imgs, rb_imgs = list(), list()
    for img in imgs:
        gray_img = np.mean(img, axis=-1)
        gray_img = np.stack([gray_img, gray_img, gray_img], axis=-1)
        gray_imgs.append(gray_img)

        rb_imgs.append(get_rb_img(img))

    imgs = np.float32(imgs) * 0.5 + 0.5
    imgs_id = np.float32(imgs_id) * 0.5 + 0.5
    gray_imgs = np.float32(gray_imgs) * 0.5 + 0.5
    rb_imgs = np.float32(rb_imgs)


    gen_imgs = np.concatenate([imgs, gray_imgs, rb_imgs, imgs_id], axis=0)

    ms = 256
    cs = 50
    rs = 20
    show_img = np.ones((r * ms + (r - 1) * rs, c * ms + (c - 1) * cs, 3), np.float32)
    cnt = 0
    for i in range(r):
        for j in range(c):
            show_img[i * (ms + rs): i * (ms + rs) + ms,
                     j * (ms + cs): j * (ms + cs) + ms, :] = gen_imgs[cnt][0]
            cnt += 1
    show_img = np.uint8(show_img * 255.)
    show_img = cv2.cvtColor(show_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('%s/%d_%d_%d.png' % (img_log, epoch, batch_i, step), show_img)


if __name__ == '__main__':

    start_time = datetime.datetime.now()

    # Adversarial loss ground truths
    valid = np.ones((conf.batch_size,) + conf.disc_patch)
    fake = np.zeros((conf.batch_size,) + conf.disc_patch)

    for epoch in range(conf.epochs):
        if conf.pretrained_flag and epoch < pre_epoch:
            continue
        dataloader = DataLoader(conf)
        for batch_i, (imgs_3D, imgs_our) in enumerate(dataloader.load_batch_train(conf.batch_size)):
            if conf.pretrained_flag and batch_i < pre_batch_i:
                continue
            # ----------------------
            #  Train Discriminators
            # ----------------------

            # Translate images to opposite domain
            fake_our = g_3D_our.predict(imgs_3D)
            fake_3D = g_our_3D.predict(imgs_our)

            # Train the discriminators (original images = real / translated = Fake)
            d_3D_loss_real = d_3D.train_on_batch(imgs_3D, valid)
            d_3D_loss_fake = d_3D.train_on_batch(fake_3D, fake)
            d_3D_loss = 0.5 * np.add(d_3D_loss_real, d_3D_loss_fake)

            d_our_loss_real = d_our.train_on_batch(imgs_our, valid)
            d_our_loss_fake = d_our.train_on_batch(fake_our, fake)
            d_our_loss = 0.5 * np.add(d_our_loss_real, d_our_loss_fake)

            # Total disciminator loss
            d_loss = 0.5 * np.add(d_3D_loss, d_our_loss)

            # ------------------
            #  Train Generators
            # ------------------
            imgs_3D_gray = np.mean(imgs_3D, axis=-1)
            imgs_3D_gray = np.reshape(imgs_3D_gray, imgs_3D_gray.shape + (1, ))

            imgs_our_gray = np.mean(imgs_our, axis=-1)
            imgs_our_gray = np.reshape(imgs_our_gray, imgs_our_gray.shape + (1, ))

            imgs_3D_r = r_model(imgs_3D)
            imgs_3D_b = b_model(imgs_3D)

            imgs_our_r = r_model(imgs_our)
            imgs_our_b = b_model(imgs_our)

            # Train the generators
            g_loss = combined.train_on_batch([imgs_3D, imgs_our],
                                             [valid, valid,
                                              imgs_3D, imgs_our,
                                              imgs_3D, imgs_our,
                                              imgs_3D_gray, imgs_our_gray,
                                              imgs_3D_r, imgs_our_r,
                                              imgs_3D_b, imgs_our_b])

            elapsed_time = datetime.datetime.now() - start_time

            # Plot the progress
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc:%3d%%] "
                 "[G loss: %05f, adv: %05f, recon: %05f, id: %05f, gray:%05f, r:%05f] time: %s " %
                  (epoch, conf.epochs, batch_i, dataloader.n_batches, d_loss[0], 100 * d_loss[1], g_loss[0],
                   np.mean(g_loss[1:3]), np.mean(g_loss[3:5]), np.mean(g_loss[5:7]),  np.mean(g_loss[7:9]),
                   np.mean(g_loss[9:13]), elapsed_time))

            # If at save interval => save generated image samples
            if step % conf.sample_interval == 0 and step != 0:
                sample_images_style(epoch, batch_i, step, dataloader)
                scalars = {
                    'discriminator/loss':   d_loss[0],
                    'discriminator/acc':    d_loss[1] * 100,
                    'generator/loss':       g_loss[0],
                    'generator/adv':        np.mean(g_loss[1: 3]),
                    'generator/recon':      np.mean(g_loss[3: 5]),
                    'generator/id':         np.mean(g_loss[5: 7]),
                    'generator/gray':       np.mean(g_loss[7: 9]),
                    'generator/rb':         np.mean(g_loss[9: 13])
                }
                for k in scalars.keys():
                    log_write.add_scalar('%s' % k, scalars[k], step)

            if step % conf.weight_interval == 0 and step != 0:
                g_3D_our.save_weights('%s/g_3D_our_%d_%d_%d.h5' % (wei_log, epoch, batch_i, step))
                g_our_3D.save_weights('%s/g_our_3D_%d_%d_%d.h5' % (wei_log, epoch, batch_i, step))
                d_3D.save_weights('%s/d_3D_%d_%d_%d.h5' % (wei_log, epoch, batch_i, step))
                d_our.save_weights('%s/d_our_%d_%d_%d.h5' % (wei_log, epoch, batch_i, step))
                combined.save_weights('%s/combined_%d_%d_%d.h5' % (wei_log, epoch, batch_i, step))
            step += 1