import tensorflow as tf
import tensorflow_addons as tfa

learning_rate = 1e-4
lr_schedule = tfa.optimizers.CyclicalLearningRate(
            initial_learning_rate=1e-4,
            maximal_learning_rate=1e-2,
            step_size=2000,
            scale_fn=lambda x: 1.,
            scale_mode="cycle",
            name="MyCyclicScheduler")
# optimizer = tf.keras.optimizers.Adam(learning_rate)
# optimizer = tf.keras.optimizers.RMSprop(learning_rate)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

batch_size = 8
mode = 'mobilenet_v1'
EPOCHS = 100

alpha = .90
gamma = 2
# loss_weight = [1000, 1]
loss_weight = 1000

ckpt_path = '20200901_EfficientPose-focal_alpha-90'