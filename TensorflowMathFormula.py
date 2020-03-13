import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mse')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float) # see formula on next line.
ys = np.array([-1.0, 0.5, 2.0, 3.5, 5.0, 6.5], dtype=float) # using the formula y = (3x + 1)/2.

model.fit(xs, ys, epochs=5)
print(model.predict([7.0])) # y = (3x + 1)/2. So, if x is 10, y should give you something near 31.

model.fit(xs, ys, epochs=20)
print(model.predict([7.0])) # y = (3x + 1)/2. So, if x is 10, y should give you something near 31.

model.fit(xs, ys, epochs=500)
print(model.predict([7.0])) # y = (3x + 1)/2. So, if x is 10, y should give you something near 31.

model.fit(xs, ys, epochs=1000)
print(model.predict([7.0])) # y = (3x + 1)/2. So, if x is 10, y should give you something near 31.
# Results should be approaching 11. 3x7=21, 21+1=22, 22/2=11