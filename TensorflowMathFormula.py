import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float) # see formula on next line.
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float) # using the formula y = 3x + 1.

model.fit(xs, ys, epochs=500)

print(model.predict([10.0])) # y = 3x + 1. So, if x is 10, y should give you something near 31.