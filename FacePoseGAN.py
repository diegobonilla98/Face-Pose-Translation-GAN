from tensorflow.keras.layers import Conv2D, MaxPool2D, Concatenate, Conv2DTranspose, Input, UpSampling2D, LeakyReLU, \
    PReLU, add, Dropout, BatchNormalization, Lambda, Activation, Dense, Flatten, GaussianNoise, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import vgg16
import tensorflow.keras.backend as K
from DataLoader import DataLoader
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


class FacePoseGAN:
    def __init__(self, is_demo=False):
        self.gf = 128
        self.image_shape = (128, 128, 3)
        self.batch_size = 8
        self.optimizer = Adam(lr=0.0002, beta_1=0.5)
        self.init = RandomNormal(mean=0.0, stddev=0.02)

        if not is_demo:
            self.data_loader = DataLoader(self.image_shape, self.batch_size)

        self.generator = self.build_generator()
        self.generator.summary()
        if is_demo:
            return
        plot_model(self.generator, to_file='generator_model.png', show_shapes=True)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer,
                                   loss_weights=[0.5], metrics=['acc'])
        self.discriminator.summary()
        plot_model(self.discriminator, to_file='discriminator_model.png', show_shapes=True)

        self.set_trainable(self.discriminator, False)
        input_tensor = Input(shape=(self.image_shape[0], self.image_shape[1], 4))
        input_tensor_dis = Input(shape=self.image_shape)
        gen = self.generator(input_tensor)
        generated = self.discriminator([gen, input_tensor_dis])
        self.adversarial = Model([input_tensor, input_tensor_dis], [gen, generated])
        self.adversarial.summary()
        self.adversarial.compile(loss=['mae', 'binary_crossentropy'], loss_weights=[10., 1.],
                                 optimizer=self.optimizer)
        plot_model(self.adversarial, to_file='adversarial_model.png', show_shapes=True)

    @staticmethod
    def set_trainable(model, state):
        model.trainable = state
        for layer in model.layers:
            layer.trainable = state

    def build_discriminator(self):
        filters = 64
        input_tensor_A = Input(shape=self.image_shape)
        input_tensor_B = Input(shape=self.image_shape)
        input_tensor = Concatenate()([input_tensor_A, input_tensor_B])

        def conv2d_block(input, filters, strides=1, bn=False):
            d = Conv2D(filters=filters, kernel_size=4, strides=strides, padding='same', kernel_initializer=self.init, use_bias=False)(input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.9)(d)
                # d = Dropout(0.3)(d)
            return d

        x = GaussianNoise(0.1)(input_tensor)
        x = conv2d_block(x, filters, bn=False)
        x = conv2d_block(x, filters, strides=2)
        x = conv2d_block(x, filters * 2)
        x = conv2d_block(x, filters * 2, strides=2)
        x = conv2d_block(x, filters * 4)
        x = conv2d_block(x, filters * 4, strides=2)
        x = conv2d_block(x, filters * 8)
        x = conv2d_block(x, filters * 8, strides=2)

        # output = Conv2D(1, kernel_size=4, padding='same', activation='sigmoid', kernel_initializer=self.init)(x)
        # x = Flatten()(x)
        x = Dense(filters * 16, kernel_initializer=self.init)(x)
        x = PReLU()(x)
        output = Dense(1, activation='sigmoid', kernel_initializer=self.init)(x)

        model = Model([input_tensor_A, input_tensor_B], output)

        return model

    @staticmethod
    def SubpixelConv2D(scale=2):
        def subpixel_shape(input_shape):
            dims = [input_shape[0],
                    None if input_shape[1] is None else input_shape[1] * scale,
                    None if input_shape[2] is None else input_shape[2] * scale,
                    int(input_shape[3] / (scale ** 2))]
            output_shape = tuple(dims)
            return output_shape

        def subpixel(x):
            return tf.depth_to_space(x, scale)

        return Lambda(subpixel, output_shape=subpixel_shape)

    def build_generator(self):
        def conv2d(layer_input, filters=16, strides=1, name=None, f_size=4):
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', name=name, kernel_initializer=self.init)(layer_input)
            d = BatchNormalization(momentum=0.9, name=name + "_bn")(d)
            d = PReLU(shared_axes=[1, 2])(d)
            d = Dropout(0.3)(d)
            return d

        def residual(layer_input, filters=16, strides=1, name=None, f_size=3):
            d = conv2d(layer_input, filters=filters, strides=strides, name=name, f_size=f_size)
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', name=name + "_2", kernel_initializer=self.init)(d)
            d = BatchNormalization(momentum=0.9, name=name + "_bn2")(d)
            d = add([d, layer_input])
            return d

        def conv2d_transpose(layer_input, filters=16, strides=1, name=None, f_size=4):
            u = Conv2D(filters, kernel_size=f_size, padding='same', kernel_initializer=self.init)(layer_input)
            u = self.SubpixelConv2D()(u)
            u = BatchNormalization(momentum=0.9, name=name + "_bn")(u)
            u = PReLU(shared_axes=[1, 2])(u)
            u = Dropout(0.3)(u)
            return u

        input_tensor = Input(shape=(self.image_shape[0], self.image_shape[1], 4))
        c1 = conv2d(input_tensor, filters=self.gf, strides=1, name="g_e1", f_size=7)
        c2 = conv2d(c1, filters=self.gf * 2, strides=2, name="g_e2", f_size=3)
        c3 = conv2d(c2, filters=self.gf * 4, strides=2, name="g_e3", f_size=3)

        r1 = residual(c3, filters=self.gf * 4, name='g_r1')
        r2 = residual(r1, self.gf * 4, name='g_r2')
        r3 = residual(r2, self.gf * 4, name='g_r3')
        r4 = residual(r3, self.gf * 4, name='g_r4')
        r5 = residual(r4, self.gf * 4, name='g_r5')
        r6 = residual(r5, self.gf * 4, name='g_r6')
        r7 = residual(r6, self.gf * 4, name='g_r7')
        r8 = residual(r7, self.gf * 4, name='g_r8')
        r9 = residual(r8, self.gf * 4, name='g_r9')

        d1 = conv2d_transpose(r9, filters=self.gf * 2, f_size=4, strides=2, name='g_d1_dc')
        d2 = conv2d_transpose(d1, filters=self.gf, f_size=4, strides=2, name='g_d2_dc')

        output_img = Conv2D(3, kernel_size=7, strides=1, padding='same', activation='tanh', kernel_initializer=self.init)(d2)

        return Model(inputs=input_tensor, outputs=output_img)

    def plot_images(self, epoch):
        image_in, mask_in, image_out = self.data_loader.load_batch(1)
        joint_input = np.concatenate([image_in, mask_in], axis=-1)
        res = self.generator.predict(joint_input)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        ax1.set_title("Input image")
        ax1.imshow((image_in[0, :, :, ::-1] + 1) * 0.5)
        ax1.axis('off')
        ax2.set_title("Condition Mask")
        ax2.imshow(mask_in[0, :, :, 0])
        ax2.axis('off')
        ax3.set_title("Output image")
        ax3.imshow((res[0, :, :, ::-1] + 1) * 0.5)
        ax3.axis('off')
        ax4.set_title("Ground Truth")
        ax4.imshow((image_out[0, :, :, ::-1] + 1) * 0.5)
        ax4.axis('off')
        plt.savefig(f'./RESULTS/images/epoch_{epoch}.jpg')
        plt.close()

    def fit(self, epochs):
        self.set_trainable(self.generator, True)
        for epoch in range(epochs):
            image_in, mask_in, image_out = self.data_loader.load_batch()
            # real = np.ones((self.batch_size, 8, 8, 1))
            real = np.random.uniform(0.8, 1.2, size=(self.batch_size, 8, 8, 1))

            joint_input = np.concatenate([image_in, mask_in], axis=-1)
            fake_X = self.generator.predict(joint_input)
            # fake = np.zeros((self.batch_size, 8, 8, 1))
            fake = np.random.uniform(0., 0.2, size=(self.batch_size, 8, 8, 1))

            self.set_trainable(self.discriminator, True)
            if np.random.rand() <= 0.1:
                real, fake = fake, real
            d_loss_true = self.discriminator.train_on_batch([image_out, image_in], real)
            d_loss_fake = self.discriminator.train_on_batch([fake_X, image_in], fake)

            real = np.random.uniform(0.8, 1.2, size=(self.batch_size, 8, 8, 1))
            self.set_trainable(self.discriminator, False)
            # real = np.ones((self.batch_size, 8, 8, 1))
            g_loss = self.adversarial.train_on_batch([joint_input, image_in], [image_out, real])

            print(f"[Epoch: {epoch}/{epochs}]\t[adv_loss: {g_loss}, d_fake: {d_loss_fake}, d_true: {d_loss_true}]")

            if epoch % 25 == 0:
                self.plot_images(epoch)
            if epoch % 100 == 0:
                self.generator.save(f'./RESULTS/weights/epoch_{epoch}.h5')
            if epoch > 500:
                self.optimizer.learning_rate.assign(0.0002 * (0.43 ** epoch))


if __name__ == '__main__':
    asr = FacePoseGAN()
    asr.fit(10_000)
