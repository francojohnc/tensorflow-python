import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

EPOCH = 1
BATCH = 32
path = 'data/localization/'
TRAIN_IMG_PATH = path + 'images/train/'
TRAIN_LABEL_PATH = path + 'annotations/train_labels.csv'
TEST_IMG_PATH = path + 'images/test/'
TEST_LABEL_PATH = path + 'annotations/test_labels.csv'


# preprocess the data to get the actual image data
def preprocess(image_name, bbox, path):
    image = tf.io.read_file(path + image_name)
    image = tf.image.decode_png(image, channels=1)
    # augment data
    image = image / 255
    return image, bbox


# add column to the CSV
columns = ['img_name', 'label', 'x1', 'y1', 'x2', 'y2']
# train data
train_df = pd.read_csv(TRAIN_LABEL_PATH, names=columns)

train_image_names = train_df['img_name'].values  # numpy (60000,)
train_bbox = train_df[['x1', 'y1', 'x2', 'y2']].values  # numpy (60000, 4)

# combine image names and bounding box into dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_names, train_bbox))
train_dataset = train_dataset.map(lambda name, box: preprocess(name, box, TRAIN_IMG_PATH))
train_dataset = train_dataset.batch(BATCH)

# test data
test_df = pd.read_csv(TEST_LABEL_PATH, names=columns)
test_image_names = test_df['img_name'].values  # numpy (60000,)
test_bbox = test_df[['x1', 'y1', 'x2', 'y2']].values  # numpy (60000, 4)
# combine image names and bounding box into dataset
test_dataset = tf.data.Dataset.from_tensor_slices((test_image_names, test_bbox))
test_dataset = test_dataset.map(lambda name, box: preprocess(name, box, TEST_IMG_PATH))
test_dataset = test_dataset.batch(BATCH)


# expect numpy data
def show_image(image, box):
    color = (255, 0, 0)
    thickness = 2
    p1 = (box[0], box[1])
    p2 = (box[2], box[3])
    image = np.squeeze(image, axis=-1)
    image = cv2.rectangle(image, p1, p2, color, thickness)
    plt.imshow(image, cmap='gray')
    plt.show()


# get the first batch
images, boxes = next(iter(train_dataset))
images = (images.numpy() * 255).astype('int32')
boxes = (boxes.numpy() * 100).astype('int32')


# show_image(images, boxes)


# structure the model
def get_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(100, 100, 1)))
    # convolution 1
    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
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
learning_rate = 1e-3  # 0.001

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['accuracy']
)

model.fit(train_dataset, batch_size=BATCH, epochs=EPOCH)


def predict():
    images, boxes = next(iter(test_dataset))
    boxes = (boxes.numpy() * 100).astype('int32')
    # predict
    outputs = model.predict(images)
    outputs = (outputs * 100).astype('int32')
    images = (images.numpy() * 255).astype('int32')
    for i in range(len(images)):
        show_image(images[i], outputs[i])


model.evaluate(test_dataset)

predict()
