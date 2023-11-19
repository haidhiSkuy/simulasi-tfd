# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# =============================================================================

import tensorflow as tf


def solution_B2():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    # NORMALIZE YOUR IMAGE HERE
    X_train = X_train / 255.
    X_test = X_test / 255.
    # DEFINE YOUR MODEL HERE
    # End with 10 Neuron Dense, activated by softmax
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28,28,1)),
        tf.keras.layers.MaxPool2D(3,3), 
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    # COMPILE MODEL HERE
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['acc'])
    # TRAIN YOUR MODEL HERE
    class Callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_acc') is not None and logs.get('val_acc') > 0.9) and (logs.get('acc') is not None and logs.get('acc') > 0.9):
                self.model.stop_training = True

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, callbacks=[Callback()])
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B2()
    model.save("model_B2.h5")
