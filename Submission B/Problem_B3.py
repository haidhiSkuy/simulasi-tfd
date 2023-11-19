# ========================================================================================
# PROBLEM B3
#
# Build a CNN based classifier for Rock-Paper-Scissors dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy AND validation_accuracy > 83%
# ========================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator


def solution_B3():
    data_url = 'https://github.com/dicodingacademy/assets/releases/download/release-rps/rps.zip'
    urllib.request.urlretrieve(data_url, 'rps.zip')
    local_file = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/')
    zip_ref.close()


    # SPLIT DATA INTO TRAINING AND VALIDATION SET(20%)
    source = "data/rps/"
    val_source = "data/rps-val" 
    val_size = 0.2
    for label in os.listdir(source):
        label_path = os.path.join(source, label)
        list_of_image = os.listdir(label_path)  
        for i,img in enumerate(list_of_image): 
            image = os.path.join(label_path,img)
            val_label_dir = os.path.join(val_source,label)
            if not os.path.exists(val_label_dir):
                os.makedirs(val_label_dir) 
            if i == int(len(list_of_image)*val_size):
                break
            os.rename(image, os.path.join(val_label_dir, img))

    TRAINING_DIR = "data/rps/"
    training_datagen = ImageDataGenerator(
        rescale=(1/255.), 
    )

    training_generator = training_datagen.flow_from_directory(
        directory=TRAINING_DIR,
        target_size=(150,150),
        class_mode='categorical',
    )

    VALIDATION_DIR = "data/rps-val"
    validation_datagen = ImageDataGenerator(
        rescale=(1/255.), 
    )

    validation_generator = validation_datagen.flow_from_directory(
        directory=VALIDATION_DIR,
        target_size=(150,150),
        class_mode='categorical',
    )


    model=tf.keras.models.Sequential([
    # YOUR CODE HERE, end with 3 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(64, kernel_size=3, input_shape=(150,150,3)), 
        tf.keras.layers.MaxPool2D(3,3), 
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    
    class Callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_acc') is not None and logs.get('val_acc') > 0.83) and (logs.get('acc') is not None and logs.get('acc') > 0.83):
                self.model.stop_training = True
    
    model.fit(training_generator, validation_data=validation_generator, epochs=10, callbacks=[Callback()])
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_B3()
    model.save("model_B3.h5")
