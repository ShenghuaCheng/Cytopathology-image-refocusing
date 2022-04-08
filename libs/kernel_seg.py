import cv2
import numpy as np

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate, Input
from keras.optimizers import Adam

def unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    return model

class KernelSeg:

    def __init__(self):
        self.input_size = (256, 256, 3)
        self.weight_path = './block380.h5'
        self.removeSmallRegion = 100
        self.removeConvex = 1.1


        self.model = self.__init_model__()
        pass


    def read_img(self, path):
        '''
        :param path: read img by path
        :return: img
        '''
        img = cv2.imread(path)
        if img is None:
            raise Exception("%s read None!!!" % path)
        if np.shape(img) != (256, 256, 3):
            raise Exception("img shape err, want (256, 256, 3)")
        return img

    def __normalization__(self, img):
        '''
        :param img:
        :return:
        '''
        return (np.float32(img) / 255. - 0.5) * 2.

    def __init_model__(self):
        '''
        :return: model
        '''
        model = unet(input_size=self.input_size)
        model.compile(optimizer=Adam(lr=0.001), loss=['categorical_crossentropy'],
                                    metrics=['categorical_accuracy'])
        if self.weight_path is not None:
            print('Load weights %s' % self.weight_path)
            model.load_weights(self.weight_path)
        return model

    def __post_process__(self, img):
        '''
        :param img: output of unet model
        :return: binary img of kernel
        '''
        img[img > 0.5] = 255
        img[img <= 0.5] = 0
        img = np.uint8(img)

        final_contours, final_areas = [], []
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            # 凸包面积
            convexHull = cv2.convexHull(contours[i])
            convexArea = cv2.contourArea(convexHull)

            # 去除小连通域
            if area > self.removeSmallRegion and convexArea / area < self.removeConvex:
                final_contours.append(contours[i])
                final_areas.append(area)
        img[:, :] = 0
        img = cv2.fillPoly(img, final_contours, 255)
        return img

    def predict(self, img):
        res = self.model.predict(img)
        return res

    def predict_path(self, path):
        img = self.read_img(path)
        res = self.__normalization__(img)
        res = self.predict(np.array([res]))
        res = self.__post_process__(res[0, :, :, 0])
        return res

if __name__ == '__main__':
    path = 'D:/paper_stage2/data_r'
    save_path = 'D:/paper_stage2/data_r/our_kernel'

    import os
    from tqdm import tqdm

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    ks = KernelSeg()

    names = open('D:/paper_stage2/data_r/txt/our.txt', 'r').readlines()
    names = [name.strip() for name in names]

    for name in tqdm(names):
        img_path = os.path.join(path, 'our', name)
        img = ks.read_img(img_path)
        res = ks.__normalization__(img)
        res = ks.predict(np.array([res]))
        res = ks.__post_process__(res[0, :, :, 0])

        res = np.stack([res, res, res], axis=-1)
        cv2.imwrite(os.path.join(save_path, name), res)


