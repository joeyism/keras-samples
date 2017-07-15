import numpy
import time
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
chars = sorted(list(set(raw_text))) # unique characters
chars_to_int = dict((c,i) for i,c in enumerate(chars)) # puts them into tuples

n_chars = len(raw_text)
n_vocab = len(chars) # no of unique characters
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

weights_filename = "weights-improvement-19-1.9035.hdf5"
model.load_weights(weights_filename)
model.compile(loss="categorical_crossentropy", optimizer="adam")
model.summary()

int_to_char = dict((i,c) for i,c in enumerate(chars)) # converts the output back to letters

start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print pattern

for i in range(1000):
    seq_in = [int_to_char[value] for value in pattern]
    x = numpy.reshape(pattern, (1, len(pattern), 1)) # puts then in an array within an array
    x = x / float(n_vocab) # normalizes
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result_letter = int_to_char[index]

    print "".join(seq_in), "".join([">",result_letter,"<"])

    pattern.append(index)
    pattern = pattern[1:len(pattern)]
    print "\n-------------\n"
    time.sleep(.5)

