# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np



def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE

    labels = []
    sentences = []
    for _,row in bbc.iterrows():
        label = row[0]
        text = row[1]
        labels.append(label)
        sentences.append(text)

    # Using "shuffle=False"
    train_size = int(training_portion*len(sentences)) 
    X_train, y_train = sentences[:train_size], labels[:train_size] 
    X_test, y_test = sentences[train_size:], labels[train_size:]

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(oov_token=oov_tok, num_words=vocab_size) 
    tokenizer.fit_on_texts(X_train)

    train_sequences = tokenizer.texts_to_sequences(X_train)
    train_pad = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type)

    validation_sequences = tokenizer.texts_to_sequences(X_test)
    validation_pad = pad_sequences(validation_sequences, maxlen=max_length, truncating=trunc_type)
    
    # You can also use Tokenizer to encode your label.
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    train_labels = np.array(label_tokenizer.texts_to_sequences(y_train))
    validation_labels = np.array(label_tokenizer.texts_to_sequences(y_test))


    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),  
        tf.keras.layers.Conv1D(128, kernel_size=3), 
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    optim = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optim, metrics=['acc'])
    
    class Callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('val_acc') is not None and logs.get('val_acc') > 0.91:
                self.model.stop_training = True

    model.fit(train_pad, 
        train_labels, 
        epochs=500, 
        validation_data=(validation_pad, validation_labels),
        callbacks=[Callback()]    
    )

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
