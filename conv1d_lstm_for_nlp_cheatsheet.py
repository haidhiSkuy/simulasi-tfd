import tensorflow as tf


############# ONE LSTM LAYER ######################
model1 = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),
        tf.keras.layers.LSTM(10),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

################## MULTIPLE LSTM LAYER ##############################
model2 = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),
        tf.keras.layers.LSTM(10, return_sequences=True),
        tf.keras.layers.LSTM(10),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

########################## CONV1D LAYER ############################
model3 = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(128, kernel_size=3),
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='softmax')
    ])


########################## MULTIPLE CONV1D LAYER ############################
model4 = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(164, kernel_size=3),
        tf.keras.layers.Conv1D(128, kernel_size=3),
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

########################## LSTM --> CONV1D LAYER ############################
model5 = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),
        tf.keras.layers.LSTM(10, return_sequences=True),
        tf.keras.layers.Conv1D(128, kernel_size=3),
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='softmax')
    ])


####################### CONV1D --> LSTM LAYER #######################
model6 = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(128, kernel_size=3),
        # tf.keras.layers.LSTM(10, return_sequences=True),
        tf.keras.layers.LSTM(10),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

