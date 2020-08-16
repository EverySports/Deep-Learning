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
        # self.features.trainable = False
        self.conv1 = Conv2D(1024, 1, padding='same')
        self.dconv1 = Conv2DTranspose(1024, 3, strides=2, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(1024, 3, 1, padding='same')
        self.conv3 = Conv2D(1024, 3, 1, padding='same')
        self.dconv2 = Conv2DTranspose(1024, 3, strides=2, padding='same')
        self.bn2 = BatchNormalization()
        self.conv4 = Conv2D(1024, 3, 1, padding='same')
        self.conv5 = Conv2D(1024, 3, 1, padding='same')

        self.heatmap_conv1 = Conv2D(512, 3, 1, padding='same')
        self.heatmap_conv2 = Conv2D(256, 3, 1, padding='same')
        self.heatmap = Conv2D(17, 3, 1, activation='sigmoid', padding='same')

        self.offset_conv1 = Conv2D(512, 3, 1, padding='same')
        self.offset_conv2 = Conv2D(256, 3, 1, padding='same')
        self.offset = Conv2D(34, 3, 1 , padding='same')
        # self.displacement_fwd = Conv2D(32, 1, 1, padding='same')
        # self.displacement_bwd = Conv2D(32, 1, 1, padding='same')

    def call(self, x):
        x = self.features(x)
        x = self.conv1(x)

        x = tf.nn.relu(x)
        x = self.dconv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.conv3(x)

        x = tf.nn.relu(x)
        x = self.dconv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = self.conv4(x)
        x = tf.nn.relu(x)
        x = self.conv5(x)
        x = tf.nn.relu(x)

        heatmap = self.heatmap_conv1(x)
        heatmap = tf.nn.relu(heatmap)
        heatmap = self.heatmap_conv2(heatmap)
        heatmap = tf.nn.relu(heatmap)
        heatmap = self.heatmap(heatmap)

        offset = self.offset_conv1(x)
        offset = tf.nn.relu(offset)
        offset = self.offset_conv2(offset)
        offset = tf.nn.relu(offset)
        offset = self.offset(offset)
        # displacement_fwd = self.displacement_fwd(x)
        # displacement_bwd = self.displacement_bwd(x)
        # return heatmap, offset, displacement_fwd, displacement_bwd
        return heatmap, offset

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