import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error
import tensorflow.keras.backend as K
from config import *

def AdaptiveSwingLoss(y_pred, y_true, alpha=2.1, omega=14, epsilon=1, theta=0.5):
    # lossMat = tf.zeros_like(y_pred)
    A = omega * (1 / (1 + (theta / epsilon) ** (alpha - y_true))) * (alpha - y_true) * (
                (theta / epsilon) ** (alpha - y_true - 1)) / epsilon
    C = theta * A - omega * tf.math.log(1 + (theta / epsilon) ** (alpha - y_true))
    C = tf.cast(C, dtype=tf.float32)
    case1_ind = tf.math.abs(y_true - y_pred) < theta
    case2_ind = tf.math.abs(y_true - y_pred) >= theta
    result_1 = omega * tf.math.log(
        1 + tf.math.abs((y_true[case1_ind] - y_pred[case1_ind]) / epsilon) ** (alpha - y_true[case1_ind]))
    result_2 = A[case2_ind] * tf.math.abs(y_true[case2_ind] - y_pred[case2_ind]) - C[case2_ind]
    result = tf.reduce_mean(result_1) + tf.reduce_mean(result_2)
    return result


def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
    weight_a = alpha * (1 - y_pred) ** gamma * targets
    weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

    return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b


def focal_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    logits = tf.math.log(y_pred / (1 - y_pred))

    loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss)

def EMD_loss(y_true, y_pred):
    cdf_ytrue = K.cast(K.cumsum(y_true, axis=-1), dtype=tf.float32)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)

def criterion(y_true, y_pred):  # Regression Loss

    regr_loss = mean_squared_error(y_true, y_pred)
    #     regr_loss = tf.keras.losses.Huber()(y_true, y_pred)
    loss = tf.reduce_mean(regr_loss)

    return loss


def criterion2(y_true, y_pred):  # Heatmap Loss

    loss = focal_loss(y_true, y_pred)

    return loss