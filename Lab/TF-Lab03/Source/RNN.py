# -----------------------------
# Keras SimpleRNN
# -----------------------------

# Libraries
from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import SimpleRNN
from keras.datasets import imdb

# Parameters
max_features = 5000
MAXLENGTH = 100
batch_size = 32
embedding_size = 50
hidden_size = 250
epochsSize = 2

# Load in IMDB Keras Data with 5000 Features
print('>>>>>>> Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'Total Train sequences')
print(len(x_test), 'Total Test sequences')

# Pad Sequence ( total num of samples for train and test )
print('>>>>>>> Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=MAXLENGTH)
x_test = sequence.pad_sequences(x_test, maxlen=MAXLENGTH)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('>>>>>>> Build RNN Model...')
# Initialize Sequential Model
RNNmodel = Sequential()

# Add layer for embedding word size and total features
RNNmodel.add(Embedding(max_features, embedding_size, input_length=MAXLENGTH))
RNNmodel.add(Dropout(0.2))

# Add Simple RNN
RNNmodel.add(SimpleRNN(input_dim=1, output_dim=25, batch_input_shape=(1, 3)))

# Add Vanilla Layer
RNNmodel.add(Dense(hidden_size))
RNNmodel.add(Dropout(0.2))

# Add Activation Function - ReLU
RNNmodel.add(Activation('relu'))

# Sqeeze into 1 Layer Output
RNNmodel.add(Dense(1))

# Use another Activation Function - Sigmoid to boil in everything
RNNmodel.add(Activation('sigmoid'))


# Compile the whole model
# Calculate loss using cross enropy
# Calculate accruacy
RNNmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Test/Validate the Model
RNNmodel.fit(x_train, y_train, batch_size=batch_size, epochs=epochsSize, validation_data=(x_test, y_test))