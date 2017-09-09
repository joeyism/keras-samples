# from https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Activation
filename = './warpeace_input.txt'

data = open(filename, 'r').read()
unique_chars = list(set(data))
no_unique_chars = len(unique_chars)

num_to_char = {i:char for i, char in enumerate(unique_chars)}
char_to_num = {char:i for i, char in enumerate(unique_chars)}

no_of_features = no_unique_chars
length_of_sequence = 20
no_of_hidden = 90
no_of_layers = 10
generate_text_length = 50
batch_size = 50

X = np.zeros((int(len(data)/length_of_sequence), length_of_sequence, no_of_features))
Y = np.zeros((int(len(data)/length_of_sequence), length_of_sequence, no_of_features))

# data generation
for i in range(0, int(len(data)/length_of_sequence)):
    X_sequence = data[i*length_of_sequence:(i+1)*length_of_sequence]
    X_sequence_in_num = [char_to_num[char] for char in X_sequence]

    input_sequence = np.zeros((length_of_sequence, no_of_features))

    for j in range(length_of_sequence):
        input_sequence[j][X_sequence_in_num[j]] = 1
    X[i] = input_sequence

    Y_sequence = data[i*length_of_sequence+1:(i+1)*length_of_sequence+1]
    Y_sequence_in_num = [char_to_num[char] for char in Y_sequence]
    output_sequence = np.zeros((length_of_sequence, no_of_features))
    for j in range(length_of_sequence):
        output_sequence[j][Y_sequence_in_num[j]] = 1
    Y[i] = output_sequence


# model creation
model = Sequential()
model.add(LSTM(no_of_hidden, input_shape=(None, no_unique_chars), return_sequences=True))
for i in range(no_of_layers - 1):
    model.add(LSTM(no_of_hidden, return_sequences = True))
model.add(TimeDistributed(Dense(no_unique_chars)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

# training
def generate_text(model, generate_text_length):
    generated_num = [np.random.randint(no_unique_chars)]
    generated_char = [num_to_char[first_generated_num[-1]]]
    X = np.zeros((1, generate_text_length, no_unique_chars))
    for i in range(generate_text_length):
        X[0][i][generated_num[-1]] = 1
        print(num_to_char[generated_num[-1]], end="")
        generated_num = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        generated_char.append(num_to_char[generated_num[-1]])
    return "".join(generated_char)

epoch = 0
while True:
    print("\n\n")
    model.fit(X,Y, batch_size = batch_size, verbose = 1, epochs = 1)
    epoch += 1
    generate_text(model, generate_text_length)
    if epoch % 10 == 0:
        model.save_weights('checkpoint_{}_epoch_{}.hdf5'.format(no_of_hidden, epoch))
