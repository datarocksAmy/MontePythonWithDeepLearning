# -----------------------------
# Keras CNN
# -----------------------------

# Libraries
from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb

# Parameters
max_features = 5000
MAXLENGTH = 100
batch_size = 32
embedding_size = 50
num_filters = 250
kernelSize = 3
hidden_size = 250
epochs = 2

# Load in IMDB Keras Data with 5000 Features
print('>>>>>>> Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

# Pad Sequence ( total num of samples for train and test )
print('>>>>>>> Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=MAXLENGTH)
x_test = sequence.pad_sequences(x_test, maxlen=MAXLENGTH)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('>>>>>>> Build CNN Model...')
# Initialize Sequential Model
cnnmodel = Sequential()

# Add layer for embedding word size and total features
cnnmodel.add(Embedding(max_features, embedding_size, input_length=MAXLENGTH))
cnnmodel.add(Dropout(0.2))

# Add CNN layer with 1D
cnnmodel.add(Conv1D(num_filters,
                 kernelSize,
                 padding='valid',
                 activation='relu',
                 strides=1))
# Add Max Pooling Layer with 1D
cnnmodel.add(GlobalMaxPooling1D())

# Add Vanilla Layer
cnnmodel.add(Dense(hidden_size))
cnnmodel.add(Dropout(0.2))

# Add Activation Function - ReLU
cnnmodel.add(Activation('relu'))

# Sqeeze into 1 Layer Output
cnnmodel.add(Dense(1))

# Use another Activation Function - Sigmoid to boil in everything
cnnmodel.add(Activation('sigmoid'))

# Compile the whole model
# Calculate loss using cross enropy
# Calculate accruacy
cnnmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Test/Validate the Model
cnnmodel.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))