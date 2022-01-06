import numpy as np
import tensorflow as tf

# load data
(train_data, train_label), (test_data, test_label) = tf.keras.datasets.mnist.load_data()

# flatten data
train_data = train_data.reshape(len(train_data), 784)
test_data = test_data.reshape(len(test_data), 784)

# normalize data
train_data = train_data / 255
test_data = test_data / 255

# create a model for simplicity no hidden layer
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=10, input_shape=(784,), activation=tf.nn.sigmoid))

# compile model
learning_rate = 1e-3
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# train
history = model.fit(train_data, train_label, batch_size=32, epochs=5)

# test
history = model.evaluate(test_data, test_label)

# predict
inputs = tf.constant([test_data[0]])

outputs = model.predict(inputs)
outputs = np.argmax(outputs)

print(outputs)
