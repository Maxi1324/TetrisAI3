from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

def model(windowsize):
    return Sequential([
        Dense(1, activation='relu', input_shape=(windowsize,20,10,1)),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(10, activation='relu'),
        Dense(5, activation='relu'),
    ])