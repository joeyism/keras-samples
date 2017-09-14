import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Activation
filename = './warpeace_input.txt'


num_to_char = np.load("num_to_char.npy").item()
char_to_num = np.load("char_to_num.npy").item()
no_unique_chars = len(num_to_char)

no_of_features = no_unique_chars

length_of_sequence = 100
no_of_hidden = 700
no_of_layers = 10
generate_text_length = 100
batch_size = 50

# model creation
model = Sequential()
model.add(LSTM(no_of_hidden, input_shape=(None, no_unique_chars), return_sequences=True))
for i in range(no_of_layers - 1):
    model.add(LSTM(no_of_hidden, return_sequences = True))
model.add(TimeDistributed(Dense(no_unique_chars)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

weights_filename = "checkpoint_700_epoch_30.hdf5"
model.load_weights(weights_filename)

def generate_text(model, generate_text_length):
    generated_num = [np.random.randint(no_unique_chars)]
    generated_char = [num_to_char[generated_num[-1]]]
    X = np.zeros((1, generate_text_length, no_unique_chars))
    for i in range(generate_text_length):
        X[0][i][generated_num[-1]] = 1
        print(num_to_char[generated_num[-1]], end="")
        generated_num = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        generated_char.append(num_to_char[generated_num[-1]])
    return "".join(generated_char)

generate_text(model, generate_text_length)
