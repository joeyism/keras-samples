import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Flatten, TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
chars = sorted(list(set(raw_text))) # unique characters
chars_to_int = dict((c,i) for i,c in enumerate(chars)) # puts them into tuples

n_chars = len(raw_text)
n_vocab = len(chars)
print "Total characters: ", n_chars
print "Total vocab: ", n_vocab

seq_length = 150
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    # for every 100 letter sequence, create an in (100 characeters) and an out (next sequence)
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]

    # puts each pair into a list
    dataX.append([chars_to_int[char] for char in seq_in])
    dataY.append(chars_to_int[seq_out])

n_patterns = len(dataX)
print "Total Patterns: ", n_patterns

X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = X/float(n_vocab) #normalize so all values are 0 to 1
y = np_utils.to_categorical(dataY) # put all values into unique points

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True, recurrent_dropout=0.2))
# --Bigger--
model.add(LSTM(256, recurrent_dropout=0.2, return_sequences = True))
model.add(LSTM(256))
model.add(Dropout(0.2))
# --Bigger--
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam")
model.summary()

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

print "fitting"
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
