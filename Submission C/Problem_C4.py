# =====================================================================================================
# PROBLEM C4
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
#
# Do not use lambda layers in your model.
#
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_C4():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')
    f = open('sarcasm.json')
    data = json.load(f)
    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    for dat in data: 
        sentences.append(dat['headline'])
        labels.append(dat['is_sarcastic'])

    X_train, y_train = sentences[:training_size], labels[:training_size] 
    X_test, y_test = sentences[training_size:], labels[training_size:]

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(oov_token=oov_tok, num_words=vocab_size) 
    tokenizer.fit_on_texts(X_train)

    train_sequences = tokenizer.texts_to_sequences(X_train)
    train_pad = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type)

    validation_sequences = tokenizer.texts_to_sequences(X_test)
    validation_pad = pad_sequences(validation_sequences, maxlen=max_length, truncating=trunc_type)

    model = tf.keras.Sequential([
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),  
        tf.keras.layers.Conv1D(128, kernel_size=3), 
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optim = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss="binary_crossentropy", optimizer=optim, metrics=['acc'])

    class Callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_acc') is not None and logs.get('val_acc') > 0.75) and (logs.get('val_acc') is not None and logs.get('acc') > 0.75):
                self.model.stop_training = True

    model.fit(
        train_pad, 
        np.array(y_train), 
        epochs=500, 
        validation_data=(validation_pad, np.array(y_test)),
        callbacks=[Callback()]    
    )           
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C4()
    model.save("model_C4.h5")
