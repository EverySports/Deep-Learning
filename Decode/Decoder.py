import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import *


def _nms(heat, kernel=3):
    hmax = K.pool2d(heat, (kernel, kernel), padding='same', pool_mode='max')
    keep = K.cast(K.equal(hmax, heat), K.floatx())
    return heat * keep


def decode_ddd(hm, offset, k, output_stride):
    hm = _nms(hm)
    hm_shape = K.shape(hm)
    offset_shape = K.shape(offset)
    batch, width, cat = hm_shape[0], hm_shape[2], hm_shape[3]

    hm_flat = K.reshape(hm, (batch, -1))
    offset_flat = K.reshape(offset, (offset_shape[0], -1, offset_shape[-1]))

    def _process_sample(args):
        _hm, _offset = args
        _scores, _inds = tf.math.top_k(_hm, k=k, sorted=True)
        _classes = K.cast(_inds % cat, 'float32')
        _inds = K.cast(_inds / cat, 'int32')
        _xs = K.cast(_inds % width, 'float32')
        _ys = K.cast(K.cast(_inds / width, 'int32'), 'float32')
        _xs *= output_stride
        _ys *= output_stride
        _xs += 4
        _ys += 4

        _detection = K.stack([_xs, _ys, _scores, _classes], -1)
        return _detection

    detections = K.map_fn(_process_sample, [hm_flat, offset_flat], dtype=K.floatx())
    return detections


def add_decoder(model, k=17, output_stride=256 / 32):
    def _decode(args):
        hm, offset = args
        return decode_ddd(hm, offset, k=k, output_stride=output_stride)

    output = Lambda(_decode)([*model.outputs])
    model = tf.keras.Model(model.input, output)
    return model