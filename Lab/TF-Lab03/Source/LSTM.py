# -----------------------------
# Keras LSTM
# -----------------------------

# Libraries
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

# Parameters
max_features = 5000
MAXLENGTH = 100
batch_size = 32
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
y_train = np.array(y_train)
y_test = np.array(y_test)

print('>>>>>>> Build LSTM Model...')
# Initialize Sequential Model
LSTMmodel = Sequential()

# Add layer for embedding word size and total features
LSTMmodel.add(Embedding(max_features, 128, input_length=MAXLENGTH))

# Add Bidirectional LSTM
LSTMmodel.add(Bidirectional(LSTM(64)))
LSTMmodel.add(Dropout(0.5))

# Sqeeze into 1 Layer Output with an activation Function - Sigmoid
LSTMmodel.add(Dense(1, activation='sigmoid'))

# Compile the whole model
# Calculate loss using cross enropy
# Calculate accruacy
LSTMmodel.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

# Test/Validate the Model
LSTMmodel.fit(x_train, y_train, batch_size=batch_size, epochs=epochsSize, validation_data=[x_test, y_test])