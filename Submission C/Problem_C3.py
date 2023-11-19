# =======================================================================================================
# PROBLEM C3
#
# Build a CNN based classifier for Cats vs Dogs dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is originally published in https://www.kaggle.com/c/dogs-vs-cats/data
#
# Desired accuracy and validation_accuracy > 72%
# ========================================================================================================

import tensorflow as tf
import urllib.request
import zipfile
import tensorflow as tf
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def solution_C3():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/cats_and_dogs.zip'
    urllib.request.urlretrieve(data_url, 'cats_and_dogs.zip')
    local_file = 'cats_and_dogs.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/')
    zip_ref.close()

    BASE_DIR = 'data/cats_and_dogs_filtered'
    train_dir = os.path.join(BASE_DIR, 'train')
    validation_dir = os.path.join(BASE_DIR, 'validation')

    train_datagen = ImageDataGenerator(
        rescale=(1./255),
        rotation_range=30,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        vertical_flip=True,
        brightness_range=[0.5,1.5],
        shear_range=0.2
    )

    # YOUR IMAGE SIZE SHOULD BE 150x150
    # Make sure you used "binary"
    train_generator =  train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(150,150),
        class_mode='binary',
    )


    validation_datagen = ImageDataGenerator(
        rescale=(1/255.), 
    )
    validation_generator =  validation_datagen.flow_from_directory(
        directory=validation_dir,
        target_size=(150,150),
        class_mode='binary',
    )

    
 

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=RMSprop(learning_rate=0.00001), 
        loss='binary_crossentropy',
        metrics=['acc']
    )
    
    class Callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_acc') is not None and logs.get('val_acc') > 0.72) and (logs.get('acc') is not None and logs.get('acc') > 0.72):
                self.model.stop_training = True
    
    model.fit(train_generator, validation_data=validation_generator, epochs=100, callbacks=[Callback()])
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C3()
    model.save("model_C3.h5")
