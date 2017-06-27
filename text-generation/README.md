# Text Generation
Taken from [Text Generation With LSTM Recurrent Neural Networks in Python with Keras](http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)

This sample uses Project Gutenberg’s Alice’s Adventures in Wonderland, by Lewis Carroll as a sample. The model reads in 100 characters, then tries to predict the next character. Since the past read words/letters could be used to predict future letters (i.e. female, past tense), a recurrent NN is used.

