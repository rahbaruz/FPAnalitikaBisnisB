import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import os
import cv2
from tensorflow.keras.regularizers import l2


jumlahclass = 6

# Load the MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

directory = r'C:\Users\baruz\OneDrive\Documents\AB B\CobaPCA\imgresize'  # alamat lengkap

images = []
for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        file_path = os.path.join(directory, filename)
        image = Image.open(file_path)
        images.append(np.array(image))


myimages = np.array(images)
print(myimages.shape)


dummy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

X = myimages
y = np.array(dummy)

# Shuffle the dataset
random_state = 42  # Set a random seed for reproducibility
#X_shuffled, y_shuffled = np.random.shuffle(X, y, random_state=random_state)

# Split the shuffled dataset into training and testing sets
test_size = 0.2  # Specify the percentage of the dataset to be used for testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

print ("======== setelah diacak")

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

print(y_test)

# Preprocess the data
x_train = x_train.reshape(-1, 100, 100, 3).astype('float32') / 255.0
x_test = x_test.reshape(-1, 100, 100, 3).astype('float32') / 255.0

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(jumlahclass, activation='softmax')) # Sesuaikan dengan class


# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')