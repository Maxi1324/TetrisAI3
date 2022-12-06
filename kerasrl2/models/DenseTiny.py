from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten


DenseTiny =  Sequential([
        Dense(1, activation='relu', input_shape=(1,20,10,1)),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(200, activation='relu'),
        Dense(100, activation='relu'),
        Dense(5, activation='relu'),
    ])