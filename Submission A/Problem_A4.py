# ==========================================================================================================
# PROBLEM A4
#
# Build and train a binary classifier for the IMDB review dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in http://ai.stanford.edu/~amaas/data/sentiment/
#
# Desired accuracy and validation_accuracy > 83%
# ===========================================================================================================

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_A4():
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    # YOUR CODE HERE
    train_data, test_data = imdb['train'], imdb['test']

    training_sentences, training_labels = [], []
    testing_sentences, testing_labels = [], []

    # DO NOT CHANGE THIS CODE
    for s, l in train_data:
        training_sentences.append(s.numpy().decode('utf8'))
        training_labels.append(l.numpy())

    for s, l in test_data:
        testing_sentences.append(s.numpy().decode('utf8'))
        testing_labels.append(l.numpy())

    # YOUR CODE HERE
    training_labels, testing_labels = np.array(training_labels), np.array(testing_labels)
    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    oov_tok = "<OOV>"

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(oov_token=oov_tok, num_words=vocab_size) 
    tokenizer.fit_on_texts(training_sentences)

    train_sequences = tokenizer.texts_to_sequences(training_sentences)
    train_pad_sequences = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_pad_sequences = pad_sequences(testing_sequences, maxlen=max_length, truncating=trunc_type)

    model = tf.keras.Sequential([
        # YOUR CODE HERE. Do not change the last layer.
        tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(32, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc']) 
    
    class Callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_acc') is not None and logs.get('val_acc') > 0.83) and (logs.get('acc') is not None and logs.get('acc') > 0.83):
                self.model.stop_training = True

    model.fit(train_pad_sequences, 
        training_labels, 
        epochs=20, 
        validation_data=(testing_pad_sequences, testing_labels),
        callbacks=[Callback()]
        
    )

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_A4()
    model.save("model_A4.h5")
