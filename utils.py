import matplotlib.pyplot as plt
import numpy as np
# import cv2
import random


def sample_images_style_rb(epoch, batch_i, step, dataloader, g_net, gray_model, r_model, b_model, img_log):
    r, c = 4, 8

    train_img_source = dataloader.load_data(domain='our', batch_size=1, is_testing=True)
    train_img_target = dataloader.load_data(domain='3D', batch_size=1, is_testing=True)

    val_img_source = dataloader.load_data(domain='our', batch_size=1, is_testing=False)
    val_img_target = dataloader.load_data(domain='3D', batch_size=1, is_testing=False)

    # Translate train_set images to target domain
    train_target_fake = g_net.predict(train_img_target)
    train_source_fake = g_net.predict(train_img_source)
    # Translate val_set images to target domain
    val_target_fake = g_net.predict(val_img_target)
    val_source_fake = g_net.predict(val_img_source)

    imgs = [train_img_target[0], train_target_fake[0], train_img_source[0], train_source_fake[0],
            val_img_target[0], val_target_fake[0], val_img_source[0], val_source_fake[0]]

    imgs = np.array(imgs)
    imgs_gra = gray_model(imgs)
    imgs_red = r_model(imgs)
    imgs_blu = b_model(imgs)

    gen_imgs = list()
    for img in imgs:
        gen_imgs.append(img)
    
    for img in imgs_gra:
        img = img[:, :, 0]
        img = np.stack([img, img, img], axis=-1)
        gen_imgs.append(img)

    for img in imgs_red:
        img = img[:, :, 0]
        img = np.stack([img, img * 0, img * 0], axis=-1)
        gen_imgs.append(img)
    
    for img in imgs_blu:
        img = img[:, :, 0]
        img = np.stack([img * 0, img * 0, img], axis=-1)
        gen_imgs.append(img)
    
    # gen_imgs = np.concatenate(gen_imgs)

    # Rescale images 0 - 1
    # gen_imgs = 0.5 * gen_imgs + 0.5

    # titles = ['train_target', 'fake', 'train_source', 'fake', 'val_target', 'fake', 'val_source', 'fake']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt])
            # axs[i, j].set_title(titles[j])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig('%s/%d_%d_%d.png' % (img_log, epoch, batch_i, step))
    plt.close()

def sample_images_style(epoch, batch_i, step, dataloader, g_net, img_log):
    r, c = 3, 8

    train_img_source = dataloader.load_data(domain='our', batch_size=1, is_testing=True)
    train_img_target = dataloader.load_data(domain='3D', batch_size=1, is_testing=True)

    val_img_source = dataloader.load_data(domain='our', batch_size=1, is_testing=False)
    val_img_target = dataloader.load_data(domain='3D', batch_size=1, is_testing=False)

    # Translate train_set images to target domain
    train_target_fake = g_net.predict(train_img_target)
    train_source_fake = g_net.predict(train_img_source)
    # Translate val_set images to target domain
    val_target_fake = g_net.predict(val_img_target)
    val_source_fake = g_net.predict(val_img_source)

    imgs = [train_img_target, train_target_fake, train_img_source, train_source_fake, 
            val_img_target, val_target_fake, val_img_source, val_source_fake]

    gen_imgs = list()
    for img in imgs:
        gen_imgs.append(img)
    
    for img in imgs:
        img_gray = rgb_to_gray(img)
        gen_imgs.append(img_gray)

    for img in imgs:
        img_red = rgb_to_color_mask(img, 0)
        gen_imgs.append(img_red)
    
    gen_imgs = np.concatenate(gen_imgs)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    # titles = ['train_target', 'fake', 'train_source', 'fake', 'val_target', 'fake', 'val_source', 'fake']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt])
            # axs[i, j].set_title(titles[j])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig('%s/%d_%d_%d.png' % (img_log, epoch, batch_i, step))
    plt.close()


def sample_images_clear_du(epoch, batch_i, step, dataloader, models, img_log):
    bl_imgs, cl_imgs = dataloader.load_data(batch_size=1, is_testing=True)
    fake_clear = models['clear_model'].predict(bl_imgs)
    gen_imgs = np.concatenate([bl_imgs , fake_clear , cl_imgs])
    gen_imgs = np.uint8((gen_imgs + 1) * 127.5)
    gen_imgs = cv2.hconcat(gen_imgs)
    gen_imgs = cv2.cvtColor(gen_imgs , cv2.COLOR_RGB2BGR)
    cv2.imwrite('%s/%d_%d_%d.png' % (img_log, epoch, batch_i, step) , gen_imgs)

def sample_images_clear_du_mix(epoch, batch_i, step, dataloader, models, img_log):
    bl1_imgs, bl2_imgs , cl_imgs = dataloader.load_data(batch_size=1, is_testing=True)
    fake_clear1 = models['clear_model'].predict(bl1_imgs)
    fake_clear2 = models['clear_model'].predict(bl2_imgs)
    gen_imgs1 = np.concatenate([bl1_imgs , fake_clear1 , cl_imgs])
    gen_imgs1 = np.uint8((gen_imgs1 + 1) * 127.5)
    gen_imgs1 = cv2.hconcat(gen_imgs1)

    gen_imgs2 = np.concatenate([bl2_imgs , fake_clear2 , cl_imgs])
    gen_imgs2 = np.uint8((gen_imgs2 + 1) * 127.5)
    gen_imgs2 = cv2.hconcat(gen_imgs2)

    gen_imgs = cv2.vconcat([gen_imgs1 , gen_imgs2])

    gen_imgs = cv2.cvtColor(gen_imgs , cv2.COLOR_RGB2BGR)
    cv2.imwrite('%s/%d_%d_%d.png' % (img_log, epoch, batch_i, step) , gen_imgs)

def sample_images_clear_l1(epoch, batch_i, step , dataloader , models , img_log):
    r, c = 2, 4

    imgs_A , imgs_Al = dataloader.load_data(domain='style_A', batch_size=1, is_testing=True)
    imgs_B , imgs_Bl = dataloader.load_data(domain='style_B', batch_size=1, is_testing=True)

    # Translate images to the other domain
    fake_B = models['g_AB'].predict(imgs_A)
    fake_A = models['g_BA'].predict(imgs_B)
    # Translate back to original domain
    reconstr_A = models['g_BA'].predict(fake_B)
    reconstr_B = models['g_AB'].predict(fake_A)

    if imgs_Al[0] is None:
        imgs_Al  = np.zeros(np.shape(imgs_A) , dtype=np.uint8)
    gen_imgs = np.concatenate([imgs_A, fake_B, imgs_Al,  reconstr_A, imgs_B, fake_A, imgs_Bl , reconstr_B])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    titles = ['Original', 'Translated', 'label', 'Reconstructed']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[j])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig('%s/%d_%d_%d.png' % (img_log, epoch, batch_i, step))
    plt.close()

def val_style(dataloader , combined, d_net, valid , fake , conf):
    imgs_target = dataloader.load_data(domain='3D', batch_size=conf.batch_size, is_testing=True)

    fake_B = models['g_AB'].predict(imgs_A)
    fake_A = models['g_BA'].predict(imgs_B)

    dA_loss_real = models['d_A'].test_on_batch(imgs_A, valid)
    dA_loss_fake = models['d_A'].test_on_batch(fake_A, fake)
    dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

    dB_loss_real = models['d_B'].test_on_batch(imgs_B, valid)
    dB_loss_fake = models['d_B'].test_on_batch(fake_B, fake)
    dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

    # Total disciminator loss
    d_loss = 0.5 * np.add(dA_loss, dB_loss)
    imgs_A_Gray = np.mean(imgs_A, axis=-1)
    imgs_B_Gray = np.mean(imgs_B, axis=-1)
    imgs_A_Gray = np.reshape(imgs_A_Gray, imgs_A_Gray.shape + (1,))
    imgs_B_Gray = np.reshape(imgs_B_Gray, imgs_B_Gray.shape + (1,))
    # test the generators
    g_loss = models['combined'].test_on_batch([imgs_A, imgs_B],
                                    [valid, valid,
                                     imgs_A, imgs_B,
                                     imgs_A, imgs_B ,
                                     imgs_A_Gray , imgs_B_Gray])
    return d_loss , g_loss

def val_gray_red_l2(dataloader , models, valid, fake, conf):
    imgs_3D = dataloader.load_data(domain='style_3D', batch_size=conf.batch_size, is_testing=True)
    imgs_our = dataloader.load_data(domain='style_our', batch_size=conf.batch_size, is_testing=True)

    fake_3D = models['g_3D_our'].predict(imgs_3D)
    fake_our = models['g_our_3D'].predict(imgs_our)

    d_3D_loss_real = models['d_3D'].test_on_batch(imgs_3D, valid)
    d_our_loss_fake = models['d_3D'].test_on_batch(fake_3D, fake)
    d_3D_loss = 0.5 * np.add(d_3D_loss_real, d_our_loss_fake)

    d_our_loss_real = models['d_our'].test_on_batch(imgs_our, valid)
    d_our_loss_fake = models['d_our'].test_on_batch(fake_our, fake)
    d_our_loss = 0.5 * np.add(d_our_loss_real, d_our_loss_fake)

    # Total disciminator loss
    d_loss = 0.5 * np.add(d_3D_loss, d_our_loss)
    imgs_3D_Gray = np.mean(imgs_3D, axis=-1)
    imgs_our_Gray = np.mean(imgs_our, axis=-1)
    imgs_3D_Gray = np.reshape(imgs_3D_Gray, imgs_3D_Gray.shape + (1,))
    imgs_our_Gray = np.reshape(imgs_our_Gray, imgs_our_Gray.shape + (1,))

    imgs_3D_Red = (np.max(imgs_3D, axis=-1) == imgs_3D[:, :, :, 0]) * imgs_3D[:, :, :, 0]
    imgs_our_Red = (np.max(imgs_our, axis=-1) == imgs_our[:, :, :, 0]) * imgs_our[:, :, :, 0]
    imgs_3D_Red = np.reshape(imgs_3D_Red, imgs_3D_Red.shape + (1,))
    imgs_our_Red = np.reshape(imgs_our_Red, imgs_our_Red.shape + (1,))
    # test the generators
    g_loss = models['combined'].test_on_batch([imgs_3D, imgs_our],
                                    [valid, valid,
                                     imgs_3D, imgs_our,
                                     imgs_3D, imgs_our,
                                     imgs_3D_Gray, imgs_our_Gray,
                                     imgs_3D_Red, imgs_our_Red])

    return d_loss, g_loss

def val_clear_du(dataloader , models , valid , fake , conf):
    bl_imgs, cl_imgs = dataloader.load_data(batch_size=conf.batch_size, is_testing=True)
    g_loss = models['combined_model'].test_on_batch(bl_imgs , [cl_imgs , valid])

    fake_clear = models['clear_model'].predict_on_batch(bl_imgs)
    d_loss_real = models['d_model'].test_on_batch(cl_imgs, valid)
    d_loss_fake = models['d_model'].test_on_batch(fake_clear, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    return g_loss , d_loss


def val_clear_du_mix(dataloader , models , valid , fake , conf):
    bl1_imgs, bl2_imgs, cl_imgs = dataloader.load_data(batch_size=conf.batch_size, is_testing=True)
    bl1_vgg = models['vgg'].predict(bl1_imgs)
    g_loss1 = models['combined_model1'].test_on_batch(bl1_imgs, [cl_imgs, valid, bl1_vgg])
    bl2_vgg = models['vgg'].predict(bl2_imgs)
    g_loss2 = models['combined_model2'].test_on_batch(bl2_imgs, [valid, bl2_vgg])

    inds = list(range(conf.batch_size))
    inds1 = random.sample(inds, int(conf.batch_size / 2))
    bl1_imgs = bl1_imgs[inds1]
    inds2 = random.sample(inds, int(conf.batch_size / 2))
    bl2_imgs = bl2_imgs[inds2]
    bl_imgs = np.concatenate([bl1_imgs, bl2_imgs], axis=0)
    fake_clear = models['clear_model'].predict_on_batch(bl_imgs)
    d_loss_real = models['d_model'].test_on_batch(cl_imgs, valid)
    d_loss_fake = models['d_model'].test_on_batch(fake_clear, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    return g_loss1 , g_loss2 , d_loss

def val_clear_l1(dataloader , models, valid , fake , conf):
    imgs_A , imgs_Al = dataloader.load_data(domain='style_A', batch_size=conf.batch_size, is_testing=True)
    imgs_B , imgs_Bl = dataloader.load_data(domain='style_B', batch_size=conf.batch_size, is_testing=True)

    fake_B = models['g_AB'].predict(imgs_A)
    fake_A = models['g_BA'].predict(imgs_B)

    dA_loss_real = models['d_A'].test_on_batch(imgs_A, valid)
    dA_loss_fake = models['d_A'].test_on_batch(fake_A, fake)
    dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

    dB_loss_real = models['d_B'].test_on_batch(imgs_B, valid)
    dB_loss_fake = models['d_B'].test_on_batch(fake_B, fake)
    dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

    # Total disciminator loss
    d_loss = 0.5 * np.add(dA_loss, dB_loss)

    imgs_A_Red = (np.max(imgs_A, axis=-1) == imgs_A[:, :, :, 0]) * imgs_A[:, :, :, 0]
    imgs_B_Red = (np.max(imgs_B, axis=-1) == imgs_B[:, :, :, 0]) * imgs_B[:, :, :, 0]
    imgs_A_Red = np.reshape(imgs_A_Red, imgs_A_Red.shape + (1,))
    imgs_B_Red = np.reshape(imgs_B_Red, imgs_B_Red.shape + (1,))
    # test the generators
    if imgs_Al[0] is not None:
        g_loss = models['combined_l'].test_on_batch([imgs_A, imgs_B],
                                        [valid, valid,
                                         imgs_A, imgs_B,
                                         imgs_A, imgs_B ,
                                         imgs_A_Red , imgs_B_Red ,
                                         imgs_Al , imgs_Bl])
    else:
        g_loss = models['combined'].test_on_batch([imgs_A, imgs_B],
                                        [valid, valid,
                                         imgs_A, imgs_B,
                                         imgs_A, imgs_B ,
                                         imgs_A_Red , imgs_B_Red])
    return d_loss , g_loss


def tmp():
    path = 'W:/paper_stage2/data/txt'
    items = ['our' , 'our_test' , 'our_train']

    # for item in items:
    item = 'our_train'
    names = list()
    with open('%s/%s.txt' % (path , item) , 'r') as tmp:
        for line in tmp:
            line = line.strip()
            line = line.replace('our' , '3D_fake')
            names.append(line)

    with open('%s/3D_fake_train.txt' % path , 'w') as tmp:
        for line in names:
            tmp.write(line + '\n')

if __name__ == '__main__':
    # tmp()
    # z = np.exp(a)
    # z[:, 0] = z[:, 0] / np.sum(z[:, 0])
    # z[:, 1] = z[:, 1] / np.sum(z[:, 1])
    # print(z)
    img_path = 'D:/paper_stage2/data/3D'
    tmp_path = 'D:/paper_stage2/data/tmp'
    import cv2
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    from mmodels.models import ColorLayer
    from tensorflow.keras.layers import Input
    from tensorflow.keras import Model
    input = Input(shape=(256, 256, 3))
    red_layer = ColorLayer(0)
    r_output = red_layer(input)
    blu_layer = ColorLayer(2)
    b_output = blu_layer(input)

    red_layer_model = Model(inputs=input, outputs=r_output)
    blu_layer_model = Model(inputs=input, outputs=b_output)

    imgs = os.listdir(img_path)
    imgs = [x for x in imgs if x.find('.png') != -1]
    imgs = imgs[: 100]
    input_imgs = list()
    for img_name in imgs:
        img = cv2.imread(os.path.join(img_path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_imgs.append(img)

    input_imgs = np.float32(input_imgs) / 255.

    routput_imgs = red_layer_model(input_imgs)
    boutput_imgs = blu_layer_model(input_imgs)

    for in_img, rout_img, bout_img, img_name in zip(input_imgs, routput_imgs, boutput_imgs, imgs):
        in_img = np.uint8(in_img * 255)
        in_img = cv2.cvtColor(in_img, cv2.COLOR_RGB2BGR)
        rout_img = np.uint8(rout_img[:, :, 0] * 255.)
        rout_img = np.stack([rout_img * 0, rout_img * 0, rout_img], axis=-1)

        bout_img = np.uint8(bout_img[:, :, 0] * 255.)
        bout_img = np.stack([bout_img, bout_img * 0, bout_img * 0], axis=-1)

        save_img = cv2.hconcat((in_img, rout_img, bout_img))
        cv2.imwrite(os.path.join(tmp_path, img_name), save_img)
    pass