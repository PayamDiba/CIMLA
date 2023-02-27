import tensorflow as tf
from tensorflow.keras import Model, Input
import tensorflow.keras.layers as layers


def build_MLP(input_shape, mid_channels, l2 = 0, dropout = None):
    """
    Multilayer Perceptron Model
    """
    inputs = Input(shape=(input_shape,))
    x = [inputs]
    if dropout:
        x += [layers.Dropout(rate = dropout)(x[-1])]
    for c in mid_channels:
        x += [layers.Dense(c, activation = tf.nn.relu, kernel_regularizer = tf.keras.regularizers.l2(l2))(x[-1])]
        #x += [layers.Dropout(rate = 0.2)(x[-1])]
    outputs = layers.Dense(1, kernel_regularizer = tf.keras.regularizers.l2(0.001))(x[-1])
    model = Model(inputs=inputs, outputs=outputs)
    return model
