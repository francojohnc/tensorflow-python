import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras


# preprocess the data to get the actual image data
def read_image(image_name, bbox, path):
    image = tf.io.read_file(path + image_name)
    image = tf.image.decode_png(image, channels=1)
    image = image / 255
    return image, bbox


# config
BATCH_SIZE = 32

TRAIN_IMG_PATH = 'data/images/train/'
TEST_IMG_PATH = 'data/images/test/'
# add column to the CSV
columns = ['img_name', 'label', 'x1', 'y1', 'x2', 'y2']
# train data
train_df = pd.read_csv('data/annotations/train_labels.csv', names=columns)
train_image_names = train_df.img_name.values  # numpy (60000,)
train_bbox = train_df[['x1', 'y1', 'x2', 'y2']].values  # numpy (60000, 4)
# combine image names and bounding box into dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_names, train_bbox))
train_dataset = train_dataset.map(lambda name, box: read_image(name, box, TRAIN_IMG_PATH))
train_dataset = train_dataset.batch(BATCH_SIZE)

# test data
test_df = pd.read_csv('data/annotations/test_labels.csv', names=columns)
test_image_names = test_df.img_name.values  # numpy (60000,)
test_bbox = test_df[['x1', 'y1', 'x2', 'y2']].values  # numpy (60000, 4)
# combine image names and bounding box into dataset
test_dataset = tf.data.Dataset.from_tensor_slices((test_image_names, test_bbox))
test_dataset = test_dataset.map(lambda name, box: read_image(name, box, TEST_IMG_PATH))
test_dataset = test_dataset.batch(BATCH_SIZE)


# expect numpy data
def show_image(image, box):
    color = (255, 0, 0)
    thickness = 2
    p1 = (box[0], box[1])
    p2 = (box[2], box[3])
    img = np.squeeze(image, axis=-1)
    img = cv2.rectangle(img, p1, p2, color, thickness)
    plt.imshow(img, cmap='gray')
    plt.show()


# get the first batch
images, boxes = next(iter(train_dataset))
images = (images.numpy() * 255).astype('int32')
boxes = (boxes.numpy() * 100).astype('int32')


# show_image(images, boxes)


# structure the model
def get_model():
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Input(shape=(100, 100, 1))
    )
    # convolution 1
    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation=tf.nn.relu,
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
    # convert data to 1D array
    model.add(tf.keras.layers.Flatten())
    # fully connected layer
    model.add(tf.keras.layers.Dense(units=4, activation=tf.nn.sigmoid))
    return model


model = get_model()

# compile model
learning_rate = 1e-3
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['accuracy']
)


def predict():
    images, boxes = next(iter(test_dataset))
    boxes = (boxes.numpy() * 100).astype('int32')
    # predict
    outputs = model.predict(images)
    outputs = (outputs * 100).astype('int32')
    images = (images.numpy() * 255).astype('int32')
    for i in range(len(images)):
        show_image(images[i], outputs[i])


model.fit(train_dataset, batch_size=BATCH_SIZE, epochs=1)
predict()
