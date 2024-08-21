from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

def neural_network():
    cnn = Sequential()
    cnn.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=[64,64,3]))
    cnn.add(MaxPooling2D(pool_size=(2,2)))

    cnn.add(Flatten())

    cnn.add(Dense(units=128, activation='relu'))
    cnn.add(Dense(units=4, activation='softmax'))

    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return cnn