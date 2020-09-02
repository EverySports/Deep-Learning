import tensorflow as tf

from tensorflow.keras.applications import MobileNet, MobileNetV2, EfficientNetB4, EfficientNetB2
from tensorflow.keras.layers import *
from tensorflow.python.keras import backend

class DepthwiseConvBlock(tf.keras.Model):
    def __init__(self, pointwise_conv_filters, alpha=1.0, depth_multiplier=1,
                          strides=(1, 1), block_id=1):
        super(DepthwiseConvBlock, self).__init__()
        self.channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
        self.pointwise_conv_filters = int(pointwise_conv_filters * alpha)
        self.depth_multiplier = depth_multiplier
        self.strides = strides
        self.block_id = block_id

        self.depth_conv = DepthwiseConv2D((3, 3),
                                   padding='same' if self.strides == (1, 1) else 'valid',
                                   depth_multiplier=self.depth_multiplier,
                                   strides=self.strides,
                                   use_bias=False,
                                   name='conv_dw_%d' % self.block_id)
        self.bn1 = BatchNormalization(
            axis=self.channel_axis, name='conv_dw_%d_bn' % self.block_id)
        self.relu1 = ReLU(6., name='conv_dw_%d_relu' % self.block_id)
        self.conv = Conv2D(
            self.pointwise_conv_filters, (1, 1),
            padding='same',
            use_bias=False,
            strides=(1, 1),
            name='conv_pw_%d' % self.block_id)
        self.bn2 = BatchNormalization(
            axis=self.channel_axis, name='conv_pw_%d_bn' % self.block_id)
        self.relu2 = ReLU(6., name='conv_pw_%d_relu' % self.block_id)

    def call(self, x):
        x = self.depth_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class DenseUnit(tf.keras.Model):
    def __init__(self, filter_out):
        super(DenseUnit, self).__init__()
        self.conv1 = DepthwiseConvBlock(filter_out)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = DepthwiseConvBlock(filter_out)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.concat1 = tf.keras.layers.Concatenate()
        self.conv3 = DepthwiseConvBlock(filter_out)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.concat2 = tf.keras.layers.Concatenate()

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.swish(x)

        h = self.conv2(x)
        h = self.bn2(h)
        h = tf.nn.swish(h)
        x = self.concat1([x, h])

        h = self.conv3(x)
        h = self.bn3(h)
        h = tf.nn.swish(h)
        x = self.concat2([x, h])
        return x

class DetectionBlock(tf.keras.Model):
    def __init__(self, filter_out):
        super(DetectionBlock, self).__init__()
        self.denseUnit1 = DenseUnit(filter_out)
        self.denseUnit2 = DenseUnit(filter_out)

    def call(self, x):
        x = self.denseUnit1(x)
        h = self.denseUnit2(x)
        return x + h

class MobileNetV1(tf.keras.Model):
    def __init__(self, model_id='mobilenet_v1', input_shape=(256, 256, 3), output_stride=16):
        assert model_id == 'mobilenet_v1'

        super(MobileNetV1, self).__init__()
        self.output_stride = output_stride
        self.features = EfficientNetB2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet',
            classifier_activation=None,
        )
        # self.features.trainable = False

        self.Dblock1 = DetectionBlock(144)
        self.concat1 = tf.keras.layers.Concatenate()
        self.Dblock2 = DetectionBlock(144)
        self.concat2 = tf.keras.layers.Concatenate()
        self.Dblock3 = DetectionBlock(144)

        self.conv1 = Conv2D(17, 1, 1, padding='same')
        self.bn1 = BatchNormalization()
        self.dconv1 = Conv2DTranspose(17, 4, 2, padding='same')
        self.dconv2 = Conv2DTranspose(17, 4, 2, padding='same')
        self.dconv3 = Conv2DTranspose(17, 4, 2, padding='same')

        # self.dconv1 = Conv2DTranspose(512, 3, strides=8, padding='same')
        # self.bn1 = BatchNormalization()
        # self.conv1 = Conv2D(256, 3, 1, padding='same')
        # self.conv2 = Conv2D(256, 3, 1, padding='same')
        # self.dconv2 = Conv2DTranspose(128, 3, strides=4, padding='same')
        # self.bn2 = BatchNormalization()
        #
        # self.heatmap_conv1 = Conv2D(128, 3, 1, padding='same')
        # self.heatmap_conv2 = Conv2D(64, 3, 1, padding='same')
        # self.heatmap = Conv2D(17, 3, 1, activation='sigmoid', padding='same')

    def call(self, x):
        h = self.features(x)

        x = self.Dblock1(h)
        x = self.concat1([x, h])

        x = self.Dblock2(x)
        x = self.concat2([x, h])

        x = self.Dblock3(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.swish(x)

        x = self.dconv1(x)
        x = tf.nn.swish(x)
        x = self.dconv2(x)
        x = tf.nn.swish(x)
        x = self.dconv3(x)
        heatmap = tf.nn.sigmoid(x)

        # x = self.dconv1(x)
        # x = self.bn1(x)
        # x = tf.nn.relu(x)
        # x = self.conv1(x)
        # x = tf.nn.relu(x)
        # x = self.conv2(x)
        #
        # x = tf.nn.relu(x)
        # x = self.dconv2(x)
        # x = self.bn2(x)
        # x = tf.nn.relu(x)
        #
        # heatmap = self.heatmap_conv1(x)
        # heatmap = tf.nn.relu(heatmap)
        # heatmap = self.heatmap_conv2(heatmap)
        # heatmap = tf.nn.relu(heatmap)
        # heatmap = self.heatmap(heatmap)

        return heatmap

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