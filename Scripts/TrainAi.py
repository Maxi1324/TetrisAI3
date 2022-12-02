import numpy as np

name = "12022022_214310"

loadedlabel = np.array(np.load("label "+name+".npy", allow_pickle=True),dtype=np.uint8)
loadeddata = np.array(np.load("data "+name+".npy", allow_pickle=True),dtype=np.uint8)



from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten,Dropout,Convolution2D,MaxPooling2D
from tensorflow.python.keras.losses import mean_squared_error; 
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD


model =  Sequential([
        Dense(1, activation='relu', input_shape=(1,20,10,1)),
        Flatten(),
        Dense(25, activation='relu'),
        Dense(25, activation='relu'),
        Dense(25, activation='relu'),
        Dense(5, activation='relu'),
    ])

print(model.summary())

model.compile(optimizer=SGD(learning_rate=1e-2),loss= mean_squared_error,metrics=['accuracy'])
model.build()

model.fit(loadeddata,loadedlabel,epochs=200,verbose=2)


print("trained")