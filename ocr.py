import os
import string

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

EPOCH = 100
BATCH = 16
PATH = 'data/captcha/'
WIDTH = 200
HEIGHT = 50
CHANNEL = 1
DIMENSION = (WIDTH, HEIGHT, CHANNEL)
LENGTH = 5 # max length of output text

characters = list(string.digits + string.ascii_lowercase)

char_to_num = tf.keras.layers.StringLookup(
    vocabulary=characters,
)
num_to_char = tf.keras.layers.StringLookup(
    oov_token='-',
    vocabulary=characters,
    invert=True
)

# data preprocessing
def preprocess(path):
    filename = tf.strings.split(path, os.path.sep)[-1]
    filename = tf.strings.split(filename, '.')[0]
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=CHANNEL)
    image = tf.image.convert_image_dtype(image, tf.float32)  # normalize
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    image = tf.transpose(image, perm=[1, 0, 2])  # rotate image
    label = tf.strings.unicode_split(filename, input_encoding="UTF-8")
    label = char_to_num(label)
    label = label - 1
    return image, label


dataset = tf.data.Dataset.list_files(PATH + '*')
dataset = dataset.map(preprocess)
dataset = dataset.batch(BATCH)

# create model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=DIMENSION))
model.add(
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation=tf.nn.relu,
        kernel_initializer="he_normal",
        padding="same",
    )
)
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation=tf.nn.relu,
        padding="same",
    )
)
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
new_shape = ((WIDTH // 4), (HEIGHT // 4) * 64)
model.add(tf.keras.layers.Reshape(target_shape=new_shape))
model.add(tf.keras.layers.Dense(
    units=64,
    activation=tf.nn.relu
))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True, dropout=0.25)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True, dropout=0.25)))
model.add(tf.keras.layers.Dense(
    units=len(characters),
    activation=tf.nn.softmax
))
model.summary()


def ctc_loss(y_true, y_pred):
    label_length = tf.ones(shape=(tf.shape(y_true)[0], 1), dtype="int32") * tf.shape(y_true)[1]
    input_length = tf.ones(shape=(tf.shape(y_pred)[0], 1), dtype="int32") * tf.shape(y_pred)[1]
    loss = tf.keras.backend.ctc_batch_cost(
        y_true=y_true,
        y_pred=y_pred,
        label_length=label_length,
        input_length=input_length
    )
    return loss


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=ctc_loss,
)
model.summary()
model.fit(dataset, batch_size=BATCH, epochs=EPOCH)

images, labels = next(iter(dataset))
outputs = model.predict(images)

input_len = np.ones(outputs.shape[0]) * outputs.shape[1]
outputs = tf.keras.backend.ctc_decode(outputs, input_length=input_len)

outputs = outputs[0][0]
outputs = outputs[:, :LENGTH]
# convert number to character
outputs = num_to_char(outputs + 1)
# join all columns character array to string byte
outputs = tf.strings.reduce_join(outputs, axis=-1)
# convert tensor to numpy
outputs = outputs.numpy()
for i in range(BATCH):
    img = images[i]
    img = img * 255  # denormalize value
    img = img.numpy().astype("uint8")  # convert image to numpy
    img = img[:, :, 0].T  # rotate image
    label = outputs[i]
    label = label.decode("utf-8")  # decode to utf-8
    plt.xlabel(label)
    plt.imshow(img, cmap="gray")
    plt.show()
