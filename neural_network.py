from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# Keras tutorial : Develop your first neural network in python step-by-step
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# Load the dataset
dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')

# Split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]

# Define the Keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the Keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the Keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10, verbose=0)

# Evaluate the Keras model
# loss, accuracy = model.evaluate(X, y, verbose=0)
# print('Accuracy: %.2f' % (accuracy*100))

# Make class predictions with the model
predictions = model.predict_classes(X)

# Summarize the first 5 cases
for i in range(5):
  print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))