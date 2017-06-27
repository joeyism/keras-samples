from keras.models import Sequential
from keras.layers import Dense
import numpy

numpy.random.seed(7)

# load data
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]


# model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


print "Compiling"
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print "Fitting"
model.fit(X,Y, epochs=150, batch_size=10)

print "Evaluating"
scores = model.evaluate(X,Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


