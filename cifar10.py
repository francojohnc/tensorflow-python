import numpy as np
import tensorflow as tf

# load data
(train_data, train_label), (test_data, test_label) = tf.keras.datasets.cifar10.load_data()
# normalize data
train_data = train_data / 255
test_data = test_data / 255
# reshape 2D to 1D array
train_label = train_label.reshape(-1, )
test_label = test_label.reshape(-1, )

# create a model
model = tf.keras.Sequential()
# convolution 1
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# convolution 2
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# convert matrix to 1 array
model.add(tf.keras.layers.Flatten())
# fully connected layer
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

# compile model
learning_rate = 1e-3
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
# train
history = model.fit(train_data, train_label, batch_size=32, epochs=2)

# test
history = model.evaluate(test_data, test_label)

# predict
inputs = tf.constant([test_data[0]])

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

outputs = model.predict(inputs)
outputs = np.argmax(outputs)
outputs = classes[outputs]

print(outputs)
