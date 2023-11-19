# =============================================================================
# PROBLEM C2
#
# Create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 91%
# =============================================================================

import tensorflow as tf


def solution_C2():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
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
            if (logs.get('val_acc') is not None and logs.get('val_acc') > 0.95) and (logs.get('acc') is not None and logs.get('acc') > 0.95):
                self.model.stop_training = True
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, callbacks=[Callback()])
    
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C2()
    model.save("model_C2.h5")
