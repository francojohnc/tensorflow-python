import tensorflow as tf

EPOCH = 1
BATCH = 32
DIMENSION = 32
path = "data/classification/images"

classes = [
    'Speed limit (20km/h)',
    'Speed limit (30km/h)',
    'Speed limit (50km/h)',
    'Speed limit (60km/h)',
    'Speed limit (70km/h)',
    'Speed limit (80km/h)',
    'End of speed limit (80km/h)',
    'Speed limit (100km/h)',
    'Speed limit (120km/h)',
    'No passing',
    'No passing for vehicles over 3.5 metric tons',
    'Right-of-way at the next intersection',
    'Priority road',
    'Yield', 'Stop',
    'No vehicles',
    'Vehicles over 3.5 metric tons prohibited',
    'No entry',
    'General caution',
    'Dangerous curve to the left',
    'Dangerous curve to the right',
    'Double curve',
    'Bumpy road',
    'Slippery road',
    'Road narrows on the right',
    'Road work', 'Traffic signals',
    'Pedestrians', 'Children crossing',
    'Bicycles crossing',
    'Beware of ice/snow',
    'Wild animals crossing',
    'End of all speed and passing limits',
    'Turn right ahead',
    'Turn left ahead',
    'Ahead only',
    'Go straight or right',
    'Go straight or left',
    'Keep right',
    'Keep left',
    'Roundabout mandatory',
    'End of no passing',
    'End of no passing by vehicles over 3.5 metric tons']

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=path,
    color_mode='grayscale',
    batch_size=BATCH,
    image_size=(DIMENSION, DIMENSION)
)


def preprocess(image, label):
    image = image / 255
    return image, label


dataset = dataset.map(preprocess)


def get_model():
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Input(shape=(DIMENSION, DIMENSION, 1))
    )
    # model.add(tf.keras.layers.Conv2D(
    #     filters=60,
    #     kernel_size=(5, 5),
    #     activation='relu'
    # ))
    # model.add(tf.keras.layers.Conv2D(
    #     filters=60,
    #     kernel_size=(5, 5),
    #     activation='relu'
    # ))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(tf.keras.layers.Conv2D(
    #     filters=30,
    #     kernel_size=(3, 3),
    #     activation='relu'
    # ))
    # model.add(tf.keras.layers.Conv2D(
    #     filters=30,
    #     kernel_size=(3, 3),
    #     activation='relu'
    # ))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(units=500, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(units=len(classes), activation='softmax'))
    return model


model = get_model()
# COMPILE MODEL
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
result = model.fit(
    dataset,
    batch_size=BATCH,
    epochs=EPOCH,
)

# predict
