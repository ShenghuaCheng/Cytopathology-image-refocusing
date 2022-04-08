from tensorflow.keras.layers import Input,  MaxPooling2D , AveragePooling2D , add , Lambda , BatchNormalization
from tensorflow.keras.layers import LeakyReLU , PReLU , UpSampling2D , Conv2D , concatenate
from tensorflow.keras.models import Model

def DUNet(conf , residual_beta = 0.5 , denses = 2):

    def mul(x, beta):
        return x * beta

    def DenseBlockR(x , channels , beta = 0.5):
        module1_out = Conv2D(channels , kernel_size=3 , strides=1 , padding='same')(x)
        module1_out = LeakyReLU(alpha=0.2)(module1_out)

        module1_out_temp = add([x , module1_out])

        module2_out = Conv2D(channels , kernel_size=3 , strides=1 , padding='same')(module1_out_temp)
        module2_out = LeakyReLU(alpha=0.2)(module2_out)

        module2_out_temp = add([x , module1_out_temp , module2_out])

        module3_out = Conv2D(channels , kernel_size=3 , strides=1 , padding='same')(module2_out_temp)
        module3_out = LeakyReLU(alpha=0.2)(module3_out)

        last_conv = Conv2D(channels , kernel_size=3 , strides=1 , padding='same')(module3_out)
        out = add([x , Lambda(mul, arguments={'beta': beta})(last_conv)])
        return out

    # Image input
    d0 = Input(shape=conf.img_shape)
    d1 = Conv2D(64 , kernel_size=9 , strides=1 , padding='same')(d0)
    d1 = PReLU()(d1)

    d2 = Conv2D(64 , kernel_size=3 , strides=1 , padding='same')(d1)
    d2 = PReLU()(d2)
    for i in range(denses):
        d2 = DenseBlockR(d2 , 64 , residual_beta)
    ds2 = AveragePooling2D(pool_size=(2 , 2) , strides=2)(d2)

    d3 = Conv2D(128 , kernel_size=3 , strides=1 , padding='same')(ds2)
    d3 = PReLU()(d3)
    for i in range(denses):
        d3 = DenseBlockR(d3 , 128 , residual_beta)
    ds3 = AveragePooling2D(pool_size=(2 , 2) , strides=2)(d3)

    d4 = Conv2D(256 , kernel_size=3 , strides=1 , padding='same')(ds3)
    d4 = PReLU()(d4)
    for i in range(denses):
        d4 = DenseBlockR(d4 , 256 , residual_beta)
    ds4 = AveragePooling2D(pool_size=(2 , 2) , strides=2)(d4)

    bottom = Conv2D(512 , kernel_size=3 , strides=1 ,padding='same')(ds4)
    bottom = PReLU()(bottom)
    bottom = Conv2D(512 , kernel_size=3 , strides=1 ,padding='same')(bottom)
    bottom = PReLU()(bottom)
    bottom = Conv2D(256 , kernel_size=3 , strides=1 ,padding='same')(bottom)
    bottom = PReLU()(bottom)

    u4 = UpSampling2D(size=(2, 2))(bottom)
    u4 = concatenate([u4 , d4], axis=-1)
    u4 = Conv2D(256 , kernel_size=3 , padding='same')(u4)
    u4 = PReLU()(u4)
    for i in range(denses):
        u4 = DenseBlockR(u4 , 256 , residual_beta)
    u4 = Conv2D(128 , kernel_size=3, padding='same')(u4)
    u4 = PReLU()(u4)

    u3 = UpSampling2D(size=(2 , 2))(u4)
    u3 = concatenate([d3 , u3] , axis = -1)
    u3 = Conv2D(128 , kernel_size=3 , padding='same')(u3)
    u3 = PReLU()(u3)
    for i in range(denses):
        u3 = DenseBlockR(u3 , 128 , residual_beta)
    u3 = Conv2D(64 , kernel_size=3, padding='same')(u3)
    u3 = PReLU()(u3)

    u2 = UpSampling2D(size=(2 , 2))(u3)
    u2 = concatenate([d2 , u2] , axis = -1)
    u2 = Conv2D(64 , kernel_size=3 , padding='same')(u2)
    u2 = PReLU()(u2)
    for i in range(denses):
        u2 = DenseBlockR(u2 , 64 , residual_beta)
    u2 = Conv2D(64 , kernel_size=3, padding='same')(u2)
    u2 = PReLU()(u2)

    u1 = Conv2D(3 , kernel_size=9 , strides=1 , padding='same' , activation='tanh')(u2)

    return Model(d0, u1)

def PatchDiscriminator(conf):
    d0 = Input(shape=conf.img_shape)

    d1 = Conv2D(64 , kernel_size=3, padding='same')(d0)
    d1 = LeakyReLU(alpha=0.2)(d1)

    d2 = Conv2D(64 , kernel_size=3 , strides=2 , padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = LeakyReLU(alpha=0.2)(d2)

    d3 = Conv2D(128 , kernel_size=3 , padding='same')(d2)
    d3 = BatchNormalization()(d3)
    d3 = LeakyReLU(alpha=0.2)(d3)

    d4 = Conv2D(128 , kernel_size=3 , strides=2 , padding='same')(d3)
    d4 = BatchNormalization()(d4)
    d4 = LeakyReLU(alpha=0.2)(d4)

    d5 = Conv2D(256 , kernel_size=3 , padding='same')(d4)
    d5 = BatchNormalization()(d5)
    d5 = LeakyReLU(alpha=0.2)(d5)

    d6 = Conv2D(256 , kernel_size=3 , strides=2 , padding='same')(d5)
    d6 = BatchNormalization()(d6)
    d6 = LeakyReLU(alpha=0.2)(d6)

    d7 = Conv2D(512 , kernel_size=3 , padding='same')(d6)
    d7 = BatchNormalization()(d7)
    d7 = LeakyReLU(alpha=0.2)(d7)

    d8 = Conv2D(256 , kernel_size=3 , strides=2 , padding='same')(d7)
    d8 = BatchNormalization()(d8)
    d8 = LeakyReLU(alpha=0.2)(d8)

    d8 = MaxPooling2D(pool_size=(4 , 4) , strides=4)(d8)

    d9 = Conv2D(1 , kernel_size=1)(d8)
    return Model(d0, d9)


