# =====================================================================================
# PROBLEM A2
#
# Build a Neural Network Model for Horse or Human Dataset.
# The test will expect it to classify binary classes.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy and validation_accuracy > 83%
# ======================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop


def solution_A2():
    data_url_1 = 'https://github.com/dicodingacademy/assets/releases/download/release-horse-or-human/horse-or-human.zip'
    urllib.request.urlretrieve(data_url_1, 'horse-or-human.zip')
    local_file = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/horse-or-human')

    data_url_2 = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/validation-horse-or-human.zip'
    urllib.request.urlretrieve(data_url_2, 'validation-horse-or-human.zip')
    local_file = 'validation-horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/validation-horse-or-human')
    zip_ref.close()

    TRAINING_DIR = 'data/horse-or-human'
    train_datagen = ImageDataGenerator(
        rescale=(1/255)
    ) 
    
    # YOUR IMAGE SIZE SHOULD BE 150x150
    train_generator = train_datagen.flow_from_directory(
        directory=TRAINING_DIR, 
        batch_size=32, 
        class_mode='binary',
        target_size=(150,150)
    ) 

    VALIDATION_DIR = 'data/validation-horse-or-human'
    val_datagen = ImageDataGenerator(
        rescale=(1/255)
    )   
    val_generator = val_datagen.flow_from_directory(
        directory=VALIDATION_DIR, 
        batch_size=32, 
        class_mode='binary',
        target_size=(150,150)
    ) 

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=3, input_shape=(150,150,3), activation='relu'), 
        tf.keras.layers.MaxPool2D(3,3), 
        tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu'), 
        tf.keras.layers.MaxPool2D(3,3),
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])

    class Callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('val_accuracy') is not None and logs.get('val_accuracy') > 0.83:
                self.model.stop_training = True
    model.fit(train_generator, epochs=15, validation_data=val_generator, callbacks=[Callback()])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_A2()
    model.save("model_A2.h5")