import numpy as np
from multiprocessing import dummy, cpu_count
from tensorflow.keras.layers import MaxPooling2D

from libs.metrics import cal_metrics

def val_metrics_stage2(dataloader, g_BS, conf=None):
    '''
    :param dataloader:
    :param g_BS:
    :param cal_batch:
    :return:
    '''
    cal_batch = conf.cal_metrics_batch_size
    random = conf.cal_metrics_random

    imgs_dicts = dataloader.load_batch_val(batch_size=cal_batch, random=random)
    imgs_B1, imgs_B1_label = imgs_dicts['imgs_B1'], imgs_dicts['imgs_B1_label']
    imgs_B2 = imgs_dicts['imgs_B2']
    if conf.detail:
        imgs_B1_det, imgs_B1_rec = imgs_dicts['imgs_B1_det'], imgs_dicts['imgs_B1_rec']
        imgs_B2_det, imgs_B2_rec = imgs_dicts['imgs_B2_det'], imgs_dicts['imgs_B2_rec']

    if conf.att_loss:
        imgs_B1 = imgs_B1[:, :, :, : 3]
    # model predict
    if conf.detail:
        fake_S1 = g_BS.predict([imgs_B1, imgs_B1_det, imgs_B1_rec])
        fake_S2 = g_BS.predict([imgs_B2, imgs_B2_det, imgs_B2_rec])
    else:
        fake_S1 = g_BS.predict(imgs_B1)
        fake_S2 = g_BS.predict(imgs_B2)

    imgs_B1L, fake_S1 = np.uint8((imgs_B1_label + 1) * 127.5), np.uint8((fake_S1 + 1) * 127.5)
    fake_S2 = np.uint8((fake_S2 + 1) * 127.5)

    def add_metrics(img_a, img_b, ssims, psnrs, vos, sds):
        me_dict = cal_metrics(img_a, img_b, choose_metrics=['ssim', 'psnr', 'sd', 'vo'])
        ssims.append(me_dict['ssim'])
        psnrs.append(me_dict['psnr'])
        vos.append(me_dict['vo'])
        sds.append(me_dict['sd'])

    ssims1, psnrs1, vos1, sds1 = [], [], [], []
    ssims2, psnrs2, vos2, sds2 = [], [], [], []
    pool = dummy.Pool(cpu_count())
    for img_BL, fake_S in zip(imgs_B1L, fake_S1):
        pool.apply_async(add_metrics, args=(fake_S, img_BL, ssims1, psnrs1, vos1, sds1))

    for fake_S in fake_S2:
        pool.apply_async(add_metrics, args=(fake_S, None, ssims2, psnrs2, vos2, sds2))

    pool.close()
    pool.join()
    return {'ssim1': np.mean(ssims1), 'psnr1': np.mean(psnrs1), 'vo1': np.mean(vos1), 'sd1': np.mean(sds1),
            'ssim2': np.mean(ssims2), 'psnr2': np.mean(psnrs2), 'vo2': np.mean(vos2), 'sd2': np.mean(sds2), }



def loss_write(log_write, tg_loss, td_loss, vg_loss, vd_loss, step, conf=None, stage1_flag=True):
    log_write.add_scalars('discriminator/d_loss', {'td_loss': td_loss[0], 'vd_loss': vd_loss[0]}, step)
    log_write.add_scalars('discriminator/acc', {'t_acc': td_loss[1] * 100, 'v_acc': vd_loss[1] * 100}, step)
    log_write.add_scalars('generator/g_loss', {'tg_loss': tg_loss[0], 'vg_loss': vg_loss[0]}, step)
    log_write.add_scalars('generator/adv', {'t_adv': np.mean(tg_loss[1:3]),
                                            'v_adv': np.mean(vg_loss[1: 3])}, step)
    log_write.add_scalars('generator/recon', {'t_recon': np.mean(tg_loss[3: 5]),
                                              'v_recon': np.mean(vg_loss[3: 5])}, step)
    log_write.add_scalars('generator/id', {'t_id': np.mean(tg_loss[5: 7]),
                                           'v_id': np.mean(vg_loss[5: 7])}, step)
    if not stage1_flag:
        log_write.add_scalars('generator/l1', {'t_l1': np.mean(tg_loss[7: 9]),
                                               'v_l1': np.mean(vg_loss[7: 9])},
                              step)
        if conf.att_loss:
            log_write.add_scalars('generator/l1_att', {'t_l1_att': np.mean(tg_loss[9: 11]),
                                                   'v_l1_att': np.mean(vg_loss[9: 11])},
                                  step)

def log_write_clear_stage1(log_write, tg_loss, td_loss, vg_loss, vd_loss, metrics, step, conf):
    '''
    train clear stage1
    :param log_write:
    :param tg_loss: train set generator loss
    :param td_loss: train set discriminator loss
    :param vg_loss: valid set
    :param vd_loss:
    :param step: current train step
    :param metrics: evalute on metrics
    :return:
    '''
    loss_write(log_write, tg_loss, td_loss, vg_loss, vd_loss, step, conf)
    log_write.add_scalar('metrics/ssim', metrics['ssim'], step)
    log_write.add_scalar('metrics/psnr', metrics['psnr'], step)
    log_write.add_scalar('metrics/vo', metrics['vo'], step)
    log_write.add_scalar('metrics/sd', metrics['sd'], step)


def log_write_clear_stage2(log_write, tg_loss, td_loss, vg_loss, vd_loss, metrics, step, conf=None):
    '''
    train clear stage 2
    :param log_write:
    :param tg_loss: train set generator loss
    :param td_loss: train set discriminator loss
    :param vg_loss: valid set
    :param vd_loss:
    :param step: current train step
    :param metrics: evalute on metrics
    :return:
    '''
    loss_write(log_write, tg_loss, td_loss, vg_loss, vd_loss, step, stage1_flag=False, conf=conf)
    log_write.add_scalar('metrics/ssim', metrics['ssim1'], step)
    log_write.add_scalar('metrics/psnr', metrics['psnr1'], step)
    log_write.add_scalars('metrics/vo', {'3D_to_3D': metrics['vo1'], 'our_to_3D': metrics['vo2']}, step)
    log_write.add_scalars('metrics/sd', {'3D_to_3D': metrics['sd1'], 'our_to_3D': metrics['sd2']}, step)


import cv2

def cs1_sample_img_nocircle(epoch, batch_i, step, dataloader, combined, img_log, conf=None):
    r = 1
    imgs_dict = dataloader.load_batch_val(batch_size=1, is_testing=True)
    imgs_B = imgs_dict['imgs_B']
    imgs_BL = imgs_dict['imgs_B_label']
    if conf.att_loss:
        imgs_B_att = imgs_dict['imgs_B_att']
        B_g_BS_in = np.concatenate([imgs_B, imgs_B_att], axis=-1)
        imgs_B_att = np.concatenate([imgs_B_att, imgs_B_att, imgs_B_att], axis=-1)
        c = 4
    else:
        B_g_BS_in = imgs_B
        c = 3
    inputs = B_g_BS_in
    if conf.detail:
        imgs_B_ker, imgs_B_rect = imgs_dict['imgs_B_det'], imgs_dict['imgs_B_rec']
        inputs = [inputs, imgs_B_ker, imgs_B_rect]
    # model predict

    if conf.adv_flag:
        if conf.att_loss:
            _, fake_S, _ = combined.predict(inputs)
        else:
            _, fake_S = combined.predict(inputs)
    else:
        if conf.att_loss:
            fake_S, _ = combined.predict(inputs)
        else:
            fake_S = combined.predict(inputs)

    if conf.att_loss:
        gen_imgs1 = np.array([imgs_B, fake_S, imgs_BL, imgs_B_att]) * 0.5 + 0.5
    else:
        gen_imgs1 = np.array([imgs_B, fake_S, imgs_BL]) * 0.5 + 0.5

    gen_imgs = np.concatenate([gen_imgs1], axis=0)

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
    show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('%s/%d_%d_%d.png' % (img_log, epoch, batch_i, step), show_img)

def cs1_sample_img(epoch, batch_i, step, dataloader, combined, img_log, conf=None):

    r = 2
    imgs_dict = dataloader.load_batch_val(batch_size=1, is_testing=True)
    imgs_B = imgs_dict['imgs_B']
    imgs_BL = imgs_dict['imgs_B_label']
    imgs_S = imgs_dict['imgs_S']
    imgs_SL = imgs_dict['imgs_S_label']
    if conf.att_loss:
        imgs_B_att = imgs_dict['imgs_B_att']
        imgs_S_att = imgs_dict['imgs_S_att']
        B_g_BS_in = np.concatenate([imgs_B, imgs_B_att], axis=-1)
        S_g_SB_in = np.concatenate([imgs_S, imgs_S_att], axis=-1)

        imgs_B_att = np.concatenate([imgs_B_att, imgs_B_att, imgs_B_att], axis=-1)
        imgs_S_att = np.concatenate([imgs_S_att, imgs_S_att, imgs_S_att], axis=-1)
        c = 5
    else:
        B_g_BS_in = imgs_B
        S_g_SB_in = imgs_S
        c = 4
    inputs = [B_g_BS_in, S_g_SB_in]
    if conf.detail:
        imgs_B_ker, imgs_B_rect = imgs_dict['imgs_B_det'], imgs_dict['imgs_B_rec']
        imgs_S_ker, imgs_S_rect = imgs_dict['imgs_S_det'], imgs_dict['imgs_S_rec']
        inputs += [imgs_B_ker, imgs_B_rect, imgs_S_ker, imgs_S_rect]
    # model predict
    _, _, \
    reconstr_B, reconstr_S, \
    _, _, \
    fake_S, fake_B, _, _ = combined.predict(inputs)

    if conf.att_loss:
        gen_imgs1 = np.array([imgs_B, fake_S, reconstr_B, imgs_BL, imgs_B_att]) * 0.5 + 0.5
        gen_imgs3 = np.array([imgs_S, fake_B, reconstr_S, imgs_SL, imgs_S_att]) * 0.5 + 0.5
    else:
        gen_imgs1 = np.array([imgs_B, fake_S, reconstr_B, imgs_BL]) * 0.5 + 0.5
        gen_imgs3 = np.array([imgs_S, fake_B, reconstr_S, imgs_SL]) * 0.5 + 0.5

    gen_imgs = np.concatenate([gen_imgs1, gen_imgs3], axis=0)

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
    show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('%s/%d_%d_%d.png' % (img_log, epoch, batch_i, step), show_img)

def cs1_val_loss_nocircle(dataloader, models, valid, fake, conf):
    imgs_dict = dataloader.load_batch_val(batch_size=conf.batch_size, is_testing=True)
    imgs_B = imgs_dict['imgs_B']
    imgs_BL = imgs_dict['imgs_B_label']
    if conf.att_loss:
        imgs_B_att = imgs_dict['imgs_B_att']
        B_g_BS_in = np.concatenate([imgs_B, imgs_B_att], axis=-1)
        imgs_B_att = np.concatenate([imgs_B_att, imgs_B_att, imgs_B_att], axis=-1)
        imgs_BL_att = ((imgs_BL * 0.5 + 0.5) * (imgs_B_att * 0.5 + 0.5) - 0.5) * 2
    else:
        B_g_BS_in = imgs_B

    g_B_input = imgs_B
    if conf.detail:
        imgs_B_ker, imgs_B_rect = imgs_dict['imgs_B_det'], imgs_dict['imgs_B_rec']
        g_B_input = [g_B_input, imgs_B_ker, imgs_B_rect]

    fake_S = models['g_BS'].predict(g_B_input)

    if conf.adv_flag:
        dS_loss_real = models['d_S'].test_on_batch(imgs_BL, valid)
        dS_loss_fake = models['d_S'].test_on_batch(fake_S, fake)
        dS_loss = 0.5 * np.add(dS_loss_real, dS_loss_fake)
    else:
        dS_loss = np.zeros((12,))

    # test the generators
    if conf.adv_flag:
        outputs = [valid, imgs_BL]
    else:
        outputs = [imgs_BL]

    if conf.att_loss:
        outputs += [imgs_BL_att]

    combine_input = [B_g_BS_in]
    if conf.detail:
        combine_input += [imgs_B_ker, imgs_B_rect]
    g_loss = models['combined'].test_on_batch(combine_input, outputs)
    if not conf.adv_flag:
        g_loss = [g_loss[0], 0, ] + g_loss[1:]
    return dS_loss, g_loss

def cs1_val_loss(dataloader, models, valid, fake, conf):
    imgs_dict = dataloader.load_batch_val(batch_size=conf.batch_size, is_testing=True)
    imgs_B = imgs_dict['imgs_B']
    imgs_BL = imgs_dict['imgs_B_label']
    imgs_S = imgs_dict['imgs_S']
    imgs_SL = imgs_dict['imgs_S_label']
    if conf.att_loss:
        imgs_B_att = imgs_dict['imgs_B_att']
        imgs_S_att = imgs_dict['imgs_S_att']
        B_g_BS_in = np.concatenate([imgs_B, imgs_B_att], axis=-1)
        S_g_SB_in = np.concatenate([imgs_S, imgs_S_att], axis=-1)

        imgs_B_att = np.concatenate([imgs_B_att, imgs_B_att, imgs_B_att], axis=-1)
        imgs_S_att = np.concatenate([imgs_S_att, imgs_S_att, imgs_S_att], axis=-1)

        imgs_BL_att = ((imgs_BL * 0.5 + 0.5) * (imgs_B_att * 0.5 + 0.5) - 0.5) * 2
        imgs_SL_att = ((imgs_SL * 0.5 + 0.5) * (imgs_S_att * 0.5 + 0.5) - 0.5) * 2
    else:
        B_g_BS_in = imgs_B
        S_g_SB_in = imgs_S

    g_B_input = imgs_B
    g_S_input = imgs_S
    if conf.detail:
        imgs_B_ker, imgs_B_rect = imgs_dict['imgs_B_det'], imgs_dict['imgs_B_rec']
        imgs_S_ker, imgs_S_rect = imgs_dict['imgs_S_det'], imgs_dict['imgs_S_rec']
        g_B_input = [g_B_input, imgs_B_ker, imgs_B_rect]
        g_S_input = [g_S_input, imgs_S_ker, imgs_S_rect]

    fake_S = models['g_BS'].predict(g_B_input)
    fake_B = models['g_SB'].predict(g_S_input)

    dB_loss_real = models['d_B'].test_on_batch(imgs_B, valid)
    dB_loss_fake = models['d_B'].test_on_batch(fake_B, fake)
    dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

    dS_loss_real = models['d_S'].test_on_batch(imgs_S, valid)
    dS_loss_fake = models['d_S'].test_on_batch(fake_S, fake)
    dS_loss = 0.5 * np.add(dS_loss_real, dS_loss_fake)

    # Total disciminator loss
    d_loss = 0.5 * np.add(dB_loss, dS_loss)

    # test the generators
    outputs = [valid, valid,  # valid_A , valid_B
                 imgs_B, imgs_S,  # reconstr_A , reconstr_B
                 imgs_B, imgs_S,  # img_A_id , img_B_id
                 imgs_BL, imgs_SL]
    if conf.att_loss:
        outputs += [imgs_BL_att, imgs_SL_att]

    combine_input = [B_g_BS_in, S_g_SB_in]
    if conf.detail:
        combine_input += [imgs_B_ker, imgs_B_rect, imgs_S_ker, imgs_S_rect]
    g_loss = models['combined'].test_on_batch(combine_input, outputs)
    return d_loss, g_loss

def cs1_val_metrics(dataloader, g_BS, conf=None):
    '''
    :param dataloader:
    :param g_BS:
    :param conf:
    :return:
    '''
    imgs_dict = dataloader.load_batch_val\
            (batch_size=conf.cal_metrics_batch_size, is_testing=True, random=conf.cal_metrics_random)
    imgs_B = imgs_dict['imgs_B']
    imgs_BL = imgs_dict['imgs_B_label']

    g_B_input = imgs_B
    if conf.detail:
        imgs_B_ker, imgs_B_rect = imgs_dict['imgs_B_det'], imgs_dict['imgs_B_rec']
        g_B_input = [g_B_input, imgs_B_ker, imgs_B_rect]

    # model predict
    fake_S = g_BS.predict(g_B_input)
    imgs_BL, fake_S = np.uint8((imgs_BL + 1) * 127.5), np.uint8((fake_S + 1) * 127.5)

    md = {}
    for k in conf.cal_metrics:
        md[k] = list()

    def add_metrics(img_a, img_b=None):
        for k in md.keys():
            md[k].append(metrics_dict[k](img_a, img_b, multi_channel=True))

    pool = dummy.Pool(cpu_count())
    for img_BL, fake_S in zip(imgs_BL, fake_S):
        pool.apply_async(add_metrics, args=(fake_S, img_BL))
    pool.close()
    pool.join()

    # get mean metrics
    for k in md.keys():
        md[k] = np.mean(md[k])

    return md
