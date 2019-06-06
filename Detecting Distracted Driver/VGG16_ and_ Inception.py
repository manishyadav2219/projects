'''
There are three models in this code.
1. VGG16
2. Inception 
'''


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D ,Convolution2D
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score
from keras.layers import Input
import keras
from keras.models import Model

#image augmentation 
#increasing 1000 images in each class
import random
import os
from skimage import io
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from skimage.util import random_noise
from skimage.transform import rotate
import cv2
# our folder path containing some images
folder_path = 'G:\state-farm-distracted-driver-detection\imgs2\c9'
# the number of file to generate
num_files_desired = 1000

# loop on all files of the folder and build a list of files paths
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
num_generated_files = 0
image_to_transform=[]
while num_generated_files <= num_files_desired:
    # random image from the folder
    image_path = random.choice(images)
    # read image as an two dimensional array of pixels
    x= cv2.imread(image_path)
    image_to_transform.append(x)
    num_generated_files=num_generated_files+1
    #print(num_generated_files)
    
img=np.array(image_to_transform)

import cv2
folder_path='G:\state-farm-distracted-driver-detection\imgs2\c9'
i=0
new_file_path = '%s/augmented_image_%i.jpg' % (folder_path, num_generated_files)
print(new_file_path)
# write image to the disk
x=32000
for i in range(1001):
    new_file_path = '%s/img_%s.jpg' % (folder_path,x)
    x=x+1
    sk.io.imsave(new_file_path, img[i])
    
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.4, 
    zoom_range = 0.3,
    horizontal_flip = True,
    validation_split=0.2) 

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    'G:\state-farm-distracted-driver-detection\imgs',
    target_size = (32, 32),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode='categorical',
    subset='training'
    )
validation_generator = train_datagen.flow_from_directory(
    'G:\state-farm-distracted-driver-detection\imgs',
    target_size = (32, 32),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode = 'categorical',
    subset='validation')

import cv2
import matplotlib.pyplot as plt
batch_size=32
from keras.callbacks import History 
history = History()

#VGG16
batch_size=32
num_val_samples=22424
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (32, 32, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])

history=model.fit_generator(train_generator, samples_per_epoch=25764, nb_epoch=10, 
                    validation_data=validation_generator, nb_val_samples=800)

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

model.save('model.h5')
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loss= model.evaluate_generator(validation_generator,steps=800)
print("Loss: " + str(loss[0]) + "     Accuracy" + str(loss[1]))


from keras.preprocessing import image
import numpy as np
from keras.models import load_model
#model = load_model('model.h5')
test_image =image.load_img('G:\state-farm-distracted-driver-detection\imgabc.jpg',target_size = (32,32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
images = np.vstack([test_image])
classes =loaded_model.predict(images)
print (classes)



#Inception
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.4,
    zoom_range = 0.3,
    horizontal_flip = True,
    validation_split=0.2) 

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator3 = train_datagen.flow_from_directory(
    'G:\state-farm-distracted-driver-detection\imgs',
    target_size = (32, 32),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode='categorical',
    subset='training'
    )
validation_generator3 = train_datagen.flow_from_directory(
      'G:\state-farm-distracted-driver-detection\imgs',
    target_size = (32, 32),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode = 'categorical',
    subset='validation')

input1 = Input(shape = (32, 32, 3))

layer_1 = Conv2D(64, (1,1), padding='same', activation='relu')(input1)
layer_1 = Conv2D(64, (3,3), padding='same', activation='relu')(layer_1)
layer_2 = Conv2D(64, (1,1), padding='same', activation='relu')(input1)
layer_2 = Conv2D(64, (5,5), padding='same', activation='relu')(layer_2)
layer_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input1)
layer_3 = Conv2D(64, (1,1), padding='same', activation='relu')(layer_3)

output = keras.layers.concatenate([layer_1, layer_2, layer_3], axis = 3)

output = Flatten()(output)
out    = Dense(10, activation='softmax')(output)

model2 = Model(inputs = input1, outputs = out)

from keras.optimizers import SGD
epochs = 10
alpha = 0.01
decay = alpha/epochs
sgd = SGD(lr=alpha, momentum=0.9, decay=decay, nesterov=False)
model2.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history=model2.fit_generator(train_generator3, samples_per_epoch=25764, nb_epoch=epochs, 
                    validation_data=validation_generator3, nb_val_samples=800)


model2.save('model2.h5')
model2_json = model2.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model2_json)
model2.save_weights("model2.h5")
print("Saved model to disk")

from keras.models import model_from_json
json_file = open('model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model2 = model_from_json(loaded_model_json)
loaded_model2.load_weights("model2.h5")
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

#predicting images using Inception
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from keras.models import Model
test_image =image.load_img('G:\state-farm-distracted-driver-detection\img5.jpg',target_size = (32,32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
classes =loaded_model2.predict(images)
print (classes)

from keras.optimizers import SGD
epochs=10
alpha = 0.01
decay = alpha/epochs
sgd = SGD(lr=alpha, momentum=0.9, decay=decay, nesterov=False)
loaded_model2.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
loss=loaded_model2.evaluate_generator(validation_generator3,steps=800)
print("Loss: " + str(loss[0]) + "     Accuracy " + str(loss[1]))

