import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import cv2

class NeuralNet(object):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    def _init_(self):
        self.training_images = x_train.reshape((6000, 28, 28)).astype('float') / 255
        self.training_targets = to_categorical(y_train)
        self.training_images = x_test.reshape((6000, 28, 28)).astype('float') / 255
        self.training_targets = to_categorical(y_test)
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(512, activation='relu'))
        self.model.add(tf.keras.layers.Dense(10, activation='softmax'))

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
        self.model.fit(self.training_images, self.training_targets, validation_splt=0.3, callback=[EarlyStopping(patience=2)],epochs=50)



# history = model.fit(x_train, y_train, epochs= 5)
def predict(self,image):
    input = cv2.resize(image, (28,28)).reshape((28,28,1)).astype('float32')/255
    return self.model.predict_classes(np.array9[input])







