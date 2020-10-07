import tensorflow as tf
import tensorflow_addons as tfa

learning_rate = 5e-5

def trianfle_fn(x):
    return 1. / (2.**(x - 1))

lr_schedule = tfa.optimizers.CyclicalLearningRate(
            initial_learning_rate=5e-5,
            maximal_learning_rate=2e-3,
            step_size=20,
            scale_fn=trianfle_fn,
            scale_mode="cycle",
            name="MyCyclicScheduler")

# optimizer = tf.keras.optimizers.Adam(learning_rate)
# optimizer = tf.keras.optimizers.RMSprop(learning_rate)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

batch_size = 8
mode = 'mobilenet_v1'
EPOCHS = 100

alpha = .25
gamma = 2
loss_weight = [1000, 1]
# loss_weight = 1000

ckpt_path = '20201005_EverySports-K_verification_alpha-25'