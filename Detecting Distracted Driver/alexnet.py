
from keras.layers import Convolution2D , MaxPooling2D , Dense ,Flatten,Dropout,Conv2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input
from keras.optimizers import SGD
import keras
from keras.models import model_from_json
from keras.layers import Activation
import matplotlib.pyplot as plt
from keras.callbacks import History 
history = History()

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2, 
    zoom_range = 0.2,
    horizontal_flip = True,
    validation_split = 0.2)

test_datagen = ImageDataGenerator(rescale = 1./255)

#Alexnet (for image size (227,227))


train_generator = train_datagen.flow_from_directory(
    'train',
    target_size = (227, 227),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode='categorical',
    subset = 'training'
    )
validation_generator = train_datagen.flow_from_directory(
    'train',
    target_size = (227, 227),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'validation')

shape = (227,227,3)

classifier = Sequential()
classifier.add(Convolution2D(96,(11,11),strides=(4,4),input_shape=shape,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(3,3),strides =(2,2),))
classifier.add(Convolution2D(256,(11,11),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(3,3),strides =(2,2)))
classifier.add(Convolution2D(384,3,3,activation = 'relu'))
classifier.add(Convolution2D(384,3,3,activation = 'relu'))
classifier.add(Convolution2D(384,3,3,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides =(2,2)))

classifier.add(Flatten())
classifier.add(Dense(4096, input_shape=shape,activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(4096, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(1000, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(10, activation = 'softmax'))
classifier.summary()

epochs = 10
alpha = 0.01
decay = alpha/epochs
sgd = SGD(lr=alpha, momentum=0.9, decay=decay, nesterov=False)
classifier.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history = classifier.fit_generator(train_generator, samples_per_epoch=17943, nb_epoch=epochs, 
                    validation_data=validation_generator, nb_val_samples=800)


jsonclassifier = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(jsonclassifier)
classifier.save_weights("model.h5")
print("Saved model to disk")

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")


import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, color='red', label='Training acc')
plt.plot(epochs, val_acc, color='green', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

from keras.preprocessing import image
import numpy as np
from keras.models import load_model
test_image =image.load_img('G:\state-farm-distracted-driver-detection\img5.jpg',target_size = (227,227))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
classes =loaded_model.predict(test_image)
print (classes)

from keras.optimizers import SGD
epochs=10
alpha = 0.01
decay = alpha/epochs
sgd = SGD(lr=alpha, momentum=0.9, decay=decay, nesterov=False)
loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
loss=loaded_model.evaluate_generator(validation_generator,steps=800)
print("Loss: " + str(loss[0]) + "     Accuracy " + str(loss[1]))


#Alexnet for input img size(77,77)
train_generator1 = train_datagen.flow_from_directory(
    'train',
    target_size = (77, 77),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode='categorical',
    subset = 'training'
    )
validation_generator1 = train_datagen.flow_from_directory(
    'train',
    target_size = (77, 77),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'validation')

shape = (77,77,3)

classifier = Sequential()

classifier.add(Conv2D(96,(5,5),strides = (2,2),input_shape=shape,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides =(2,2)))
classifier.add(Conv2D(256,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides =(2,2)))
classifier.add(Conv2D(384,3,3,activation = 'relu'))
classifier.add(Conv2D(384,3,3,activation = 'relu'))
classifier.add(Conv2D(384,3,3,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides =(2,2)))

classifier.add(Flatten())
classifier.add(Dense(4096, input_shape=shape,activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(4096, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(1000, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(10, activation = 'softmax'))
classifier.summary()


epochs = 10
alpha = 0.01
decay = alpha/epochs
sgd = SGD(lr=alpha, momentum=0.9, decay=decay, nesterov=False)
classifier.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history=classifier.fit_generator(train_generator1, samples_per_epoch=25764, nb_epoch=epochs, 
                    validation_data=validation_generator1, nb_val_samples=800)

jsonclassifier = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(jsonclassifier)
classifier.save_weights("model.h5")
print("Saved model to disk")

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model2 = model_from_json(loaded_model_json)

loaded_model2.load_weights("model.h5")
print("Loaded model from disk")

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, color='red', label='Training acc')
plt.plot(epochs, val_acc, color='green', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

from keras.preprocessing import image
import numpy as np
test_image =image.load_img('G:\state-farm-distracted-driver-detection\img5.jpg',target_size = (227,227))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
classes =loaded_model2.predict(test_image)
print (classes)

from keras.optimizers import SGD
epochs=10
alpha = 0.01
decay = alpha/epochs
sgd = SGD(lr=alpha, momentum=0.9, decay=decay, nesterov=False)
loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
loss=loaded_model2.evaluate_generator(validation_generator,steps=800)
print("Loss: " + str(loss[0]) + "     Accuracy " + str(loss[1]))