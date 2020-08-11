import tensorflow as tf

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import *

class MobileNetV1(tf.keras.Model):
    def __init__(self, model_id='mobilenet_v1', input_shape=(256, 256, 3), output_stride=16):
        assert model_id == 'mobilenet_v1'

        super(MobileNetV1, self).__init__()
        self.output_stride = output_stride
        self.features = MobileNet(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet',
            classifier_activation=None,
        )
        self.features.trainable = False
        self.conv = Conv2D(1024, 1, padding='same')
        self.dconv = Conv2DTranspose(1024, 3, strides=2, padding='same')

        self.heatmap = Conv2D(17, 1, 1, activation='sigmoid', padding='same')
        self.offset = Conv2D(34, 1, 1 , padding='same')
        self.displacement_fwd = Conv2D(32, 1, 1, padding='same')
        self.displacement_bwd = Conv2D(32, 1, 1, padding='same')

    def call(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.dconv(x)

        heatmap = self.heatmap(x)
        offset = self.offset(x)
        displacement_fwd = self.displacement_fwd(x)
        displacement_bwd = self.displacement_bwd(x)
        return heatmap, offset, displacement_fwd, displacement_bwd

if __name__ == '__main__':
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    metrics = ['mse']

    model = MobileNetV1()
    model.trainable = False

    inputs = tf.keras.Input(shape=(256,256,3))
    outputs = model(inputs)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizer,
        loss=loss_object,
        metrics=metrics,
    )

    print(model.summary())