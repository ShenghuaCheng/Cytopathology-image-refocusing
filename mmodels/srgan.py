#!/usr/bin/env python
# title           :Network.py
# description     :Architecture file(Generator and Discriminator)
# author          :Deepak Birla
# date            :2018/10/30
# usage           :from Network import Generator, Discriminator
# python_version  :3.5.4

# Modules
# from keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D
# from keras.layers.core import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.layers import add


# Residual block
def res_block_gen(model, kernal_size, filters, strides):
    gen = model

    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)

    model = add([gen, model])

    return model


def up_sampling_block(model, kernal_size, filters, strides):
    # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
    # Even we can have our own function for deconvolution (i.e one made in Utils.py)
    # model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = UpSampling2D(size=2)(model)
    model = LeakyReLU(alpha=0.2)(model)

    return model


def discriminator_block(model, filters, kernel_size, strides):
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    return model


# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
def sr_generator(conf):

        gen_input = Input(shape=conf.img_shape)

        model = Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(gen_input)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
            model)

        gen_model = model

        # Using 16 Residual Blocks
        for index in range(16):
            model = res_block_gen(model, 3, 64, 1)

        model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(model)
        model = BatchNormalization(momentum=0.5)(model)
        model = add([gen_model, model])

        # Using 2 UpSampling Blocks
        # for index in range(2):
            # model = up_sampling_block(model, 3, 256, 1)

        model = Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(model)
        model = Activation('tanh')(model)

        generator_model = Model(inputs=gen_input, outputs=model)

        return generator_model


# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
def sr_discriminator(conf):

        dis_input = Input(shape=conf.img_shape[: 2] + (3, ))

        model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(dis_input)
        model = LeakyReLU(alpha=0.2)(model)

        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)

        # model = Flatten()(model)
        # model = Dense(1024)(model)
        # model = LeakyReLU(alpha=0.2)(model)

        # model = Dense(1)(model)

        model = Conv2D(1, kernel_size=3, strides=1, padding='same')(model)
        model = Activation('sigmoid')(model)

        discriminator_model = Model(inputs=dis_input, outputs=model)

        return discriminator_model


# from tensorflow.keras import backend as K
from tensorflow.keras.layers import ZeroPadding2D, MaxPooling2D, Dropout, concatenate, AveragePooling2D
from mmodels.customer_layers import Scale
def DenseUNet(conf, nb_dense_block=4, growth_rate=48, nb_filter=96, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, weights_path=None,
              args=None):
    '''Instantiate the DenseNet 161 architecture,
        # Arguments
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            classes: optional number of classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis

    concat_axis = 3
    img_input = Input(shape=conf.img_shape, name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 96
    nb_layers = [6, 12, 36, 24]  # For DenseNet-161
    box = []
    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(nb_filter, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    box.append(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        box.append(x)
        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
    box.append(x)

    up0 = UpSampling2D(size=(2,2))(x)
    conv_up0 = Conv2D(768, (3, 3), padding="same", kernel_initializer="normal", name = "conv_up0")(up0)
    bn_up0 = BatchNormalization(name = "bn_up0")(conv_up0)
    ac_up0 = Activation('relu', name='ac_up0')(bn_up0)

    up1 = UpSampling2D(size=(2,2))(ac_up0)
    conv_up1 = Conv2D(384, (3, 3), padding="same", kernel_initializer="normal", name = "conv_up1")(up1)
    bn_up1 = BatchNormalization(name = "bn_up1")(conv_up1)
    ac_up1 = Activation('relu', name='ac_up1')(bn_up1)

    up2 = UpSampling2D(size=(2,2))(ac_up1)
    conv_up2 = Conv2D(96, (3, 3), padding="same", kernel_initializer="normal", name = "conv_up2")(up2)
    bn_up2 = BatchNormalization(name = "bn_up2")(conv_up2)
    ac_up2 = Activation('relu', name='ac_up2')(bn_up2)

    up3 = UpSampling2D(size=(2,2))(ac_up2)
    conv_up3 = Conv2D(96, (3, 3), padding="same", kernel_initializer="normal", name = "conv_up3")(up3)
    bn_up3 = BatchNormalization(name = "bn_up3")(conv_up3)
    ac_up3 = Activation('relu', name='ac_up3')(bn_up3)

    up4 = UpSampling2D(size=(2, 2))(ac_up3)
    conv_up4 = Conv2D(64, (3, 3), padding="same", kernel_initializer="normal", name="conv_up4")(up4)
    conv_up4 = Dropout(rate=0.3)(conv_up4)
    bn_up4 = BatchNormalization(name="bn_up4")(conv_up4)
    ac_up4 = Activation('relu', name='ac_up4')(bn_up4)

    x = Conv2D(3, (1,1), padding="same", kernel_initializer="normal", name="dense167classifer")(ac_up4)
    x = Activation('tanh')(x)
    model = Model(img_input, x, name='denseu161')


    return model

def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv2D(inter_channel, (1, 1), name=conv_name_base+'_x1', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Conv2D(nb_filter, (3, 3), name=conv_name_base+'_x2', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = concatenate([concat_feat, x], axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter