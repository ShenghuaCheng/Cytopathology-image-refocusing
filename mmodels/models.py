from tensorflow.keras.layers import Input,  Dropout, Concatenate, AveragePooling2D, add
from tensorflow.keras.layers import LeakyReLU, PReLU, UpSampling2D, Conv2D, concatenate, Lambda, Layer, ReLU
from tensorflow.keras.models import Model
from mmodels.keras_tools import InstanceNormalization
from tensorflow.keras import backend, layers, models, utils
import tensorflow as tf
import tensorflow.keras.backend as K
import os


def build_generator_style(conf):
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
    d0 = Input(shape=conf.img_shape)

    # Downsampling
    d1 = conv2d(d0, conf.gf)
    d2 = conv2d(d1, conf.gf * 2)
    d3 = conv2d(d2, conf.gf * 4)
    d4 = conv2d(d3, conf.gf * 8)

    # Upsampling
    u1 = deconv2d(d4, d3, conf.gf * 4)
    u2 = deconv2d(u1, d2, conf.gf * 2)
    u3 = deconv2d(u2, d1, conf.gf)

    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
    return Model(d0, output_img)

def build_generator_clear(conf):
    """U-Net Generator"""

    def conv2d(layer_input, in_channel, filters, f_size=4):

        d = Conv2D(in_channel, kernel_size=3, strides=1, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = add([layer_input, d])

        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d

    def deconv2d(layer_input, skip_input, in_channel , filters, f_size=4, dropout_rate=0):
        d = Conv2D(in_channel, kernel_size=3, strides=1, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = add([layer_input, d])

        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(d)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=conf.img_shape)

    # Downsampling
    d1 = conv2d(d0, 3, conf.gf)
    d2 = conv2d(d1, conf.gf, conf.gf * 2)
    d3 = conv2d(d2, conf.gf * 2, conf.gf * 4)
    d4 = conv2d(d3, conf.gf * 4, conf.gf * 8)

    # Upsampling
    u1 = deconv2d(d4, d3, conf.gf * 8, conf.gf * 4)
    u2 = deconv2d(u1, d2, conf.gf * 8, conf.gf * 2)
    u3 = deconv2d(u2, d1, conf.gf * 4, conf.gf)

    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
    return Model(d0, output_img)

def mul(x, beta):
    return x * beta

def DenseBlockR1(x, channels, beta = 0.5):
    module1_out = Conv2D(channels, kernel_size=3, strides=1 , padding='same')(x)
    module1_out = LeakyReLU(alpha=0.2)(module1_out)

    module1_out_temp = add([x , module1_out])

    module2_out = Conv2D(channels , kernel_size=3 , strides=1 , padding='same')(module1_out_temp)
    module2_out = LeakyReLU(alpha=0.2)(module2_out)

    module2_out_temp = add([x , module1_out_temp , module2_out])

    module3_out = Conv2D(channels , kernel_size=3 , strides=1 , padding='same')(module2_out_temp)
    module3_out = LeakyReLU(alpha=0.2)(module3_out)

    last_conv = Conv2D(channels , kernel_size=3 , strides=1 , padding='same')(module3_out)
    out = add([x, Lambda(mul, arguments={'beta': beta})(last_conv)])
    return out

def build_generator_clear_dense(conf, residual_beta=0.5):

    denses = conf.denseblocks

    def conv2d(layer_input, in_channel, filters, f_size=4):

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

    def deconv2d(layer_input, skip_input, in_channel, filters, f_size=4, dropout_rate=0):
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

    # Image input
    d0 = Input(shape=conf.img_shape)

    # Downsampling
    d1 = conv2d(d0, conf.channels, conf.gf)

    d2 = conv2d(d1, conf.gf, conf.gf * 2)
    d3 = conv2d(d2, conf.gf * 2, conf.gf * 4)
    d4 = conv2d(d3, conf.gf * 4, conf.gf * 8)

    # Upsampling
    u1 = deconv2d(d4, d3, conf.gf * 8, conf.gf * 4)
    u2 = deconv2d(u1, d2, conf.gf * 8, conf.gf * 2)
    u3 = deconv2d(u2, d1, conf.gf * 4, conf.gf)

    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

    return Model(d0, output_img)


def build_discriminator_cyclegan(conf):

    def d_layer(layer_input, filters, f_size=4, strides=2, normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d

    img = Input(shape=conf.img_shape[: 2] + (3, ))

    d1 = d_layer(img, conf.df, normalization=False)
    d2 = d_layer(d1, conf.df * 2)
    d3 = d_layer(d2, conf.df * 4)
    d4 = d_layer(d3, conf.df * 8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model(img, [validity, ])

def build_discriminator(conf):

    def d_layer(layer_input, filters, f_size=4, strides=2, normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d

    img = Input(shape=conf.img_shape[: 2] + (3, ))

    d1 = d_layer(img, conf.df, normalization=False)
    d2 = d_layer(d1, conf.df * 2, f_size=3, strides=1)
    d2 = d_layer(d2, conf.df * 2)

    d3 = d_layer(d2, conf.df * 4, f_size=3, strides=1)
    d3 = d_layer(d3, conf.df * 4)

    d4 = d_layer(d3, conf.df * 8, f_size=3, strides=1)
    d4 = d_layer(d4, conf.df * 8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model(img, [validity, ])

class GrayLayer(Layer):

    def __init__(self, **kwargs):
        super(GrayLayer, self).__init__(**kwargs)

    def call(self, x):
        output = Lambda(lambda x: K.mean(x, axis=-1))(x)
        output = tf.expand_dims(output, -1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0: 3] + (1, ))


def construct_Color_Model(color_layer):
    input = Input(shape=(256, 256, 3))
    output = color_layer(input)
    return Model(input, output)

class ColorLayer(Layer):

    def __init__(self, color_channel, **kwargs):
        super(ColorLayer, self).__init__(**kwargs)



        
        self.color_channel = color_channel

    def __color__(self, x):
        # formulate from (-1, 1) to (0, 1)
        x = tf.add(tf.multiply(x, 0.5), 0.5)

        exp_x = tf.exp(x)
        sum_x = K.sum(exp_x, axis=-1)
        sum_x = K.stack([sum_x, sum_x, sum_x], axis=-1)
        x = tf.divide(exp_x, sum_x)
        x1 = tf.equal(K.max(x, axis=3), x[:, :, :, self.color_channel])
        x2 = tf.cast(x1, tf.float32)
        x3 = tf.multiply(x2, x[:, :, :, self.color_channel])
        x4 = tf.multiply(tf.subtract(x3, 1. / 3), 3. / 2)
        x5 = tf.clip_by_value(x4, 0, 1)
        x6 = tf.multiply(x5, 1 / (K.max(x5) - K.min(x5)))
        x7 = tf.clip_by_value(x6, 0, 1)

        return x7

    def call(self, x):
        output = Lambda(self.__color__)(x)
        output = tf.expand_dims(output, -1)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0: 3] + (1, )


def VGG16(include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000,
          weight_path=None,
          ):

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    cx = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(cx)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    inputs = img_input
    # Create model.
    cmodel = models.Model(inputs, cx, name='cvgg16')
    model = models.Model(inputs, x, name='vgg16')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = None
        else:
            weights_path = weight_path
        print('vgg load weight : ', weight_path)
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return cmodel


def build_generator_clear_complex(conf, residual_beta=0.5, denses=1):

    # Image input
    d0 = Input(shape=conf.img_shape)
    d1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(d0)
    d1 = PReLU()(d1)

    d2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(d1)
    d2 = PReLU()(d2)
    ds2 = AveragePooling2D(pool_size=(2, 2), strides=2)(d2)

    d3 = Conv2D(128 , kernel_size=3 , strides=1 , padding='same')(ds2)
    d3 = PReLU()(d3)
    ds3 = AveragePooling2D(pool_size=(2 , 2) , strides=2)(d3)

    d4 = Conv2D(256 , kernel_size=3 , strides=1 , padding='same')(ds3)
    d4 = PReLU()(d4)
    ds4 = AveragePooling2D(pool_size=(2 , 2) , strides=2)(d4)

    bottom = Conv2D(512, kernel_size=3, strides=1 ,padding='same')(ds4)
    bottom = PReLU()(bottom)
    bottom = Conv2D(512, kernel_size=3, strides=1 ,padding='same')(bottom)
    bottom = PReLU()(bottom)
    bottom = Conv2D(256, kernel_size=3, strides=1 ,padding='same')(bottom)
    bottom = PReLU()(bottom)

    u4 = UpSampling2D(size=(2, 2))(bottom)
    u4 = concatenate([u4 , d4], axis=-1)
    u4 = Conv2D(256 , kernel_size=3 , padding='same')(u4)
    u4 = PReLU()(u4)

    u4 = Conv2D(128 , kernel_size=3, padding='same')(u4)
    u4 = PReLU()(u4)
    u3 = UpSampling2D(size=(2 , 2))(u4)
    u3 = concatenate([d3 , u3] , axis = -1)
    u3 = Conv2D(128 , kernel_size=3 , padding='same')(u3)
    u3 = PReLU()(u3)

    u3 = Conv2D(64 , kernel_size=3, padding='same')(u3)
    u3 = PReLU()(u3)
    u2 = UpSampling2D(size=(2 , 2))(u3)
    u2 = concatenate([d2 , u2] , axis = -1)
    u2 = Conv2D(64 , kernel_size=3 , padding='same')(u2)
    u2 = PReLU()(u2)

    u2 = Conv2D(64, kernel_size=3, padding='same')(u2)
    u2 = PReLU()(u2)
    u1 = Conv2D(3, kernel_size=3, strides=1, padding='same', activation='tanh')(u2)

    return Model(d0, u1)


# if __name__ == '__main__':
#
#     red_layer = ColorLayer(0)
#     fake_target_red = red_layer(fake_target)