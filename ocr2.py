import os
import string

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

EPOCH = 100
BATCH = 16
PATH = 'data/captcha/'
WIDTH = 200
HEIGHT = 50
CHANNEL = 1
DIMENSION = (WIDTH, HEIGHT, CHANNEL)
LENGTH = 5 # max length of output text

img_width = 200
img_height = 50

downsample_factor = 4


# Get list of all the images
characters = list(string.digits + string.ascii_lowercase)

char_to_num = layers.StringLookup(
    vocabulary=characters
)
num_to_char = layers.StringLookup(
    oov_token='-',
    vocabulary=char_to_num.get_vocabulary(),
    invert=True
)


# data preprocessing
def preprocess(path):
    filename = tf.strings.split(path, os.path.sep)[-1]
    filename = tf.strings.split(filename, '.')[0]
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=CHANNEL)
    # image = image / 255
    image = tf.image.convert_image_dtype(image, tf.float32)  # normalize
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    image = tf.transpose(image, perm=[1, 0, 2])  # rotate image
    label = tf.strings.unicode_split(filename, input_encoding="UTF-8")
    label = char_to_num(label)
    # label = label - 1
    return image, label

dataset = tf.data.Dataset.list_files(PATH + '*')
dataset = dataset.map(preprocess)
dataset = dataset.batch(BATCH)

print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)
print("Characters present: ", num_to_char.get_vocabulary())

_, ax = plt.subplots(4, 4, figsize=(10, 5))
for images, labels in dataset.take(1):
    for i in range(BATCH):
        img = (images[i] * 255).numpy().astype("uint8")
        label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")
        ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")
plt.show()


def build_model():
    input = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)
    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    # Output layer
    output = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)
    model = keras.models.Model(
        inputs=input, outputs=output, name="ocr_model_v1"
    )
    return model


# Get the model
model = build_model()
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
    loss=ctc_loss
)
# Train the model
history = model.fit(
    dataset,
    epochs=EPOCH,
)

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :LENGTH]
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

#  Let's check results on some validation samples
for images, labels in dataset.take(1):
    preds = model.predict(images)
    pred_texts = decode_batch_predictions(preds)

    orig_texts = []
    for label in labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)

    _, ax = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(len(pred_texts)):
        img = (images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img = img.T
        title = f"Prediction: {pred_texts[i]}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")
plt.show()
