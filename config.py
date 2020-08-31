import tensorflow as tf

learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate)

batch_size = 8
mode = 'mobilenet_v1'
EPOCHS = 1

alpha = .90
gamma = 2
loss_weight = [1000, 1]

ckpt_path = '20200816_more_deep_layer'