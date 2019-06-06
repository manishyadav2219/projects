from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#creating CNN
classifier = Sequential()

#1
classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3),activation = 'relu'))

#2
classifier.add(MaxPooling2D(pool_size=(2,2),strides =2))

#1
classifier.add(Convolution2D(32,3,3, activation = 'relu'))

#2
classifier.add(MaxPooling2D(pool_size=(2,2),strides =2))
#3 
classifier.add(Flatten())

#5
classifier.add(Dense(output_dim = 128 ,activation ='relu'))
classifier.add(Dense(output_dim =1,activation = 'sigmoid'))

classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy' ,metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

trainset = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

testset = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        trainset,
        steps_per_epoch=8000,
        epochs=5,
        validation_data=testset,
        validation_steps=2000)

from keras.preprocessing import image
import numpy as np

testimage = image.load_img('dataset/images.jfif',target_size = (64,64))
testimage = image.img_to_array(testimage)
testimage = np.expand_dims(testimage,axis=0)
result = classifier.predict(testimage)
trainset.class_indices