import numpy as np
import tensorflow as tf
from show_image import show_image

BATCH_SIZE = 32
EPOCH = 10

# load data return numpy array
# train data (50000, 32, 32, 3)
# train label (50000, 1)
# test data (10000, 32, 32, 3)
# test label (10000, 1)
(train_data, train_label), (test_data, test_label) = tf.keras.datasets.cifar10.load_data()
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# normalize data
train_data = train_data / 255
test_data = test_data / 255
# reshape label from 2D to 1D array
train_label = train_label.reshape(-1, )
test_label = test_label.reshape(-1, )

# create a model
model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Input(shape=(32, 32, 3))
)
# convolution 1
model.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    padding="same",
    activation=tf.nn.relu
))
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
history = model.fit(train_data, train_label, batch_size=BATCH_SIZE, epochs=EPOCH)

# test
history = model.evaluate(test_data, test_label)


def predict():
    outputs = model.predict(test_data)
    for image, label in zip(test_data, outputs):
        label = classes[np.argmax(label)]
        show_image(image, label)


# predict
predict()
