from tensorflow.keras.layers import Input,  Dropout, Concatenate, AveragePooling2D, add, Lambda
from tensorflow.keras.layers import LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Model
from mmodels.keras_tools import InstanceNormalization

import numpy as np

import tensorflow as tf

def mul(x, beta):
    return x * beta

def DenseBlockR1(x, channels, beta = 0.5):

    module1_out = Conv2D(channels, kernel_size=3, strides=1, padding='same')(x)
    module1_out = LeakyReLU(alpha=0.2)(module1_out)

    module1_out_temp = add([x, module1_out])

    module2_out = Conv2D(channels, kernel_size=3, strides=1, padding='same')(module1_out_temp)
    module2_out = LeakyReLU(alpha=0.2)(module2_out)

    last_conv = Conv2D(channels, kernel_size=3, strides=1, padding='same')(module2_out)
    out = add([x, Lambda(mul, arguments={'beta': beta})(last_conv)])
    return out

def conv2d(layer_input, in_channel, filters, f_size=4, denses=2, residual_beta=0.5):

    d = Conv2D(in_channel, kernel_size=3, strides=1, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    d = add([layer_input, d])

    #  insert DenseBlock
    for k in range(denses):
        d = DenseBlockR1(d, in_channel, beta=residual_beta)

    """Layers used during downsampling"""
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = InstanceNormalization()(d)
    return d

def deconv2d(layer_input, skip_input, in_channel, filters, f_size=4, dropout_rate=0, denses=2, residual_beta=0.5):

    layer_input = Conv2D(in_channel, kernel_size=3, strides=1, padding='same')(layer_input)  # 减小特征尺度，降低内存

    d = Conv2D(in_channel, kernel_size=3, strides=1, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    d = add([layer_input, d])

    #  insert DenseBlock
    for k in range(denses):
        d = DenseBlockR1(d, in_channel, beta=residual_beta)

    """Layers used during upsampling"""
    u = UpSampling2D(size=2)(d)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
    if dropout_rate:
        u = Dropout(dropout_rate)(u)
    u = InstanceNormalization()(u)
    u = Concatenate()([u, skip_input])
    return u

def build_encoder(conf, residual_beta=0.5):

    denses = conf.denseblocks

    # Image input
    d0 = Input(shape=conf.img_shape)

    # Downsampling
    d1 = conv2d(d0, conf.channels, conf.gf, denses=denses, residual_beta=residual_beta)

    d2 = conv2d(d1, conf.gf, conf.gf * 2, denses=denses, residual_beta=residual_beta)
    d3 = conv2d(d2, conf.gf * 2, conf.gf * 4, denses=denses, residual_beta=residual_beta)
    d4 = conv2d(d3, conf.gf * 4, conf.gf * 8, denses=denses, residual_beta=residual_beta)

    # Upsampling
    u1 = deconv2d(d4, d3, conf.gf * 8, conf.gf * 4, denses=denses, residual_beta=residual_beta)
    u2 = deconv2d(u1, d2, conf.gf * 8, conf.gf * 2, denses=denses, residual_beta=residual_beta)
    u3 = deconv2d(u2, d1, conf.gf * 4, conf.gf, denses=denses, residual_beta=residual_beta)

    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

    return Model(d0, [output_img, d4, d3, d2, d1])


def build_generator_multi(conf, bottom_shape, residual_beta=0.5):
    denses = conf.denseblocks

    def encoder(d0):
        # Downsampling
        d1 = conv2d(d0, conf.channels, conf.gf, denses=denses, residual_beta=residual_beta)

        d2 = conv2d(d1, conf.gf, conf.gf * 2, denses=denses, residual_beta=residual_beta)
        d3 = conv2d(d2, conf.gf * 2, conf.gf * 4, denses=denses, residual_beta=residual_beta)
        d4 = conv2d(d3, conf.gf * 4, conf.gf * 8, denses=denses, residual_beta=residual_beta)
        return d1, d2, d3, d4

    def decoder(d1, d2, d3, d4):
        # Upsampling
        u1 = deconv2d(d4, d3, conf.gf * 8, conf.gf * 4, denses=denses, residual_beta=residual_beta)
        u2 = deconv2d(u1, d2, conf.gf * 8, conf.gf * 2, denses=denses, residual_beta=residual_beta)
        u3 = deconv2d(u2, d1, conf.gf * 4, conf.gf, denses=denses, residual_beta=residual_beta)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
        return output_img

    # Image input
    d0 = Input(shape=conf.img_shape)

    d1, d2, d3, o_d4 = encoder(d0)

    # concat bottom feature
    bottom_feature = Input(bottom_shape)
    d4 = tf.concat([o_d4, bottom_feature], axis=-1)


    d1_in = Input(d1.shape[1:])
    d2_in = Input(d2.shape[1:])
    d3_in = Input(d3.shape[1:])
    d4_in = Input(d4.shape[1:])
    output = decoder(d1_in, d2_in, d3_in, d4_in)
    decoder_model = Model([d1_in, d2_in, d3_in, d4_in], output)

    output_img = decoder_model([d1, d2, d3, d4])

    return Model([d0, bottom_feature], output_img), decoder_model

def build_generator_clear_dense(conf, residual_beta=0.5, bottom_shape=None, bottom_ouput=False):

    denses = conf.denseblocks

    # Image input
    d0 = Input(shape=conf.img_shape)

    # Downsampling
    d1 = conv2d(d0, conf.channels, conf.gf, denses=denses, residual_beta=residual_beta)

    d2 = conv2d(d1, conf.gf, conf.gf * 2, denses=denses, residual_beta=residual_beta)
    d3 = conv2d(d2, conf.gf * 2, conf.gf * 4, denses=denses, residual_beta=residual_beta)
    d4 = conv2d(d3, conf.gf * 4, conf.gf * 8, denses=denses, residual_beta=residual_beta)

    # concat bottom feature
    if bottom_shape is not None:
        bottom_feature = Input(bottom_shape)
        d4 = tf.concat([d4, bottom_feature], axis=-1)

    # Upsampling
    u1 = deconv2d(d4, d3, conf.gf * 8, conf.gf * 4, denses=denses, residual_beta=residual_beta)
    u2 = deconv2d(u1, d2, conf.gf * 8, conf.gf * 2, denses=denses, residual_beta=residual_beta)
    u3 = deconv2d(u2, d1, conf.gf * 4, conf.gf, denses=denses, residual_beta=residual_beta)

    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

    if bottom_ouput:
        return Model(d0, [output_img, d4])
    else:
        if bottom_shape is not None:
            return Model([d0, bottom_feature], output_img)
        else:
            return Model(d0, output_img)

def build_generator_base(shape, gf, bottom_shape=None, bottom_ouput=False):
    """U-Net Generator"""

    def conv2d(layer_input, filters, f_size=4):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):

        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=shape)

    # Downsampling
    d1 = conv2d(d0, gf)
    d2 = conv2d(d1, gf * 2)
    d3 = conv2d(d2, gf * 4)
    d4 = conv2d(d3, gf * 8)

    # concat bottom feature
    if bottom_shape is not None:
        bottom_feature = Input(bottom_shape)
        d4 = tf.concat([d4, bottom_feature], axis=-1)

    # Upsampling
    u1 = deconv2d(d4, d3, gf * 4)
    u2 = deconv2d(u1, d2, gf * 2)
    u3 = deconv2d(u2, d1, gf)

    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
    if bottom_ouput:
        return Model(d0, [output_img, d4])
    else:
        if bottom_shape is not None:
            return Model([d0, bottom_feature], output_img)
        else:
            return Model(d0, output_img)


def __split_layer__(input, rows, cols):
    '''
    :param input:
    :param rows:
    :param cols:
    :return:
    '''
    splits = []
    x_splits = tf.split(input, rows, axis=1)
    for x_split in x_splits:
        y_splits = tf.split(x_split, cols, axis=2)
        splits += y_splits
    return splits

def __merge_layer__(input, rows, cols):
    '''
    :param input:
    :param rows:
    :param cols:
    :return:
    '''
    splits = []
    for row in range(rows):
        x_split = tf.concat(input[row * cols: (row + 1) * cols], axis=2)
        splits.append(x_split)
    splits = tf.concat(splits, axis=1)
    return splits


def build_multi_scale_v3(conf):
    '''
    :param conf:  config of net
    :return:
    '''

    model_256 = build_generator_clear_dense(conf)
    model_512 = build_generator_clear_dense(conf)

    input_256 = Input(shape=conf.img_shape)
    input_512 = UpSampling2D(size=2)(input_256)

    splits_input_512 = __split_layer__(input_512, 2, 2)
    outputs_512 = []
    for split_input_512 in splits_input_512:
        outputs_512.append(model_512(split_input_512))
    output_512 = __merge_layer__(outputs_512, 2, 2)
    output_512 = AveragePooling2D(strides=(2, 2))(output_512)

    input = tf.add(output_512, input_256)

    output = model_256(input)
    return Model(input_256, output)


def build_multi_scale_v8(conf):
    '''
    :param conf:  config of net
    :return:
    '''

    model_scale = build_generator_clear_dense(conf, bottom_ouput=True)

    scale_nums = conf.scale_num
    inputs_scale = Input(shape=(scale_nums, ) + conf.img_shape)
    inputs_scale_info = Input(shape=(scale_nums, 5))
    bottom_features = []
    for ind in range(scale_nums):
        _, bottom_feature = model_scale(inputs_scale[:, ind, :, :, :])
        bottom_features.append(bottom_feature)
    bottom_feature = tf.concat(bottom_features, axis=-1)

    model_512 = build_generator_clear_dense(conf, bottom_ouput=True)

    input_256 = Input(shape=conf.img_shape)
    input_512 = UpSampling2D(size=2)(input_256)

    splits_input_512 = __split_layer__(input_512, 2, 2)
    outputs_512 = []
    for split_input_512 in splits_input_512:
        output, _ = model_512(split_input_512)
        outputs_512.append(output)
    output_512 = __merge_layer__(outputs_512, 2, 2)
    output_512 = AveragePooling2D(strides=(2, 2))(output_512)

    model_256 = build_generator_clear_dense(conf, bottom_shape=bottom_feature.shape[1:])

    input = tf.add(output_512, input_256)

    output = model_256([input, bottom_feature])
    return Model([input_256, inputs_scale, inputs_scale_info], output)


# for output test result
def build_multi_scale_output(conf):
    '''
    :param conf:
    :return:  output kernel reconstruct and split reconstruct
    '''
    model_scale = build_encoder(conf)

    scale_nums = conf.scale_num
    inputs_scale = Input(shape=(scale_nums,) + conf.img_shape)
    inputs_scale_info = Input(shape=(scale_nums, 5))
    d4s, d3s, d2s, d1s = [], [], [], []
    for ind in range(scale_nums):
        _, d4, d3, d2, d1 = model_scale(inputs_scale[:, ind, :, :, :])
        d4s.append(d4)
        d3s.append(d3)
        d2s.append(d2)
        d1s.append(d1)
    bottom_feature = tf.concat(d4s, axis=-1)

    model_512 = build_generator_clear_dense(conf, bottom_ouput=True)

    input_256 = Input(shape=conf.img_shape)
    input_512 = UpSampling2D(size=2)(input_256)

    splits_input_512 = __split_layer__(input_512, 2, 2)
    outputs_512 = []
    for split_input_512 in splits_input_512:
        output, _ = model_512(split_input_512)
        outputs_512.append(output)
    output_512 = __merge_layer__(outputs_512, 2, 2)
    output_512 = AveragePooling2D(strides=(2, 2))(output_512)

    model_256, decoder_model = build_generator_multi(conf, bottom_feature.shape[1:])
    # model_256 = build_generator_clear_dense(conf, bottom_shape=bottom_feature.shape[1:])

    # output reconstruct kernel
    rec_ks = []
    for i in range(len(d4s)):
        d4 = tf.concat([d4s[i], d4s[i], d4s[i], d4s[i]], axis=-1)
        rec_k = decoder_model([d1s[i], d2s[i], d3s[i], d4])
        rec_ks.append(rec_k)

    input = tf.add(output_512, input_256)

    outputs = model_256([input, bottom_feature])
    return Model([input_256, inputs_scale, inputs_scale_info], [outputs, output_512, rec_ks])

if __name__ == '__main__':
    pass



