'''
DETECTING DISTRACTED DRIVER
In this code I have written my own model as a combination of inception v3 and alexnet , 
it giveg an accuracy of 96-97% . It is a good model 
for saving time and getting a good accuracy also as the images are reduced to size of (77,77).
Best results are obtained for image size of (227,227).
'''


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input
import keras
import numpy as np
import pandas as pd

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2, 
    zoom_range = 0.2,
    horizontal_flip = True,
    validation_split = 0.2)

test_datagen = ImageDataGenerator(rescale = 1./255)



train_generator = train_datagen.flow_from_directory(
    'train',
    target_size = (77, 77),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode='categorical',
    subset = 'training'
    )
validation_generator = train_datagen.flow_from_directory(
    'train',
    target_size = (77, 77),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'validation')

input1 = Input(shape = (77, 77, 3))

classifier = Sequential()

layer1 = Conv2D(96,(3,3),strides = (2,2),padding='same',activation = 'relu')(input1)
layer1 = Conv2D(96,(3,3),strides = (2,2),padding='same',activation = 'relu')(layer1)
layer1 = MaxPooling2D(pool_size=(2,2),padding='same',strides =(2,2))(layer1)
layer1 = Conv2D(256,(3,3),activation = 'relu',padding='same')(layer1)
layer1 = Conv2D(384,(3,3),activation = 'relu',padding='same')(layer1)
layer1= Conv2D(384,(3,3),activation = 'relu',padding='same')(layer1)
layer1 = MaxPooling2D(pool_size=(2,2),strides =(1,1),padding='same')(layer1)
layer_2 = Conv2D(64, (1,1), padding='same', activation='relu')(input1)
layer_2 = Conv2D(64, (3,1),strides = (2,2), padding='same', activation='relu')(layer_2)
layer_2 = Conv2D(64, (1,3),strides = (2,2), padding='same', activation='relu')(layer_2)
layer_2 = MaxPooling2D((3,3), strides=(2,2), padding='same')(layer_2)
layer_3 = MaxPooling2D((3,3), strides=(2,2), padding='same')(input1)
layer_3 = Conv2D(64, (1,1), padding='same', activation='relu')(layer_3)
layer_3 = Conv2D(64, (3,3),strides = (2,2), padding='same', activation='relu')(layer_3)
layer_3 = Conv2D(64, (3,3),strides = (2,2), padding='same', activation='relu')(layer_3)

output_layer = keras.layers.concatenate([layer1, layer_2, layer_3], axis = 3)

output = Flatten()(output_layer)
out1 =Dense(512,activation = 'relu')(output)
out2 = Dense(370, activation = 'relu')(out1)
out    = Dense(10, activation='softmax')(out2)
model = Model(inputs = input1, outputs = out)

from keras.optimizers import SGD
epochs = 10
alpha = 0.01
decay = alpha/epochs
sgd = SGD(lr=alpha, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit_generator(train_generator, samples_per_epoch=17943, nb_epoch=epochs, 
                    validation_data=validation_generator, nb_val_samples=800)

model.save('model.h5')
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")


from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loss= model.evaluate_generator(validation_generator,steps=800)
print("Loss: " + str(loss[0]) + "     Accuracy" + str(loss[1]))


from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt

#model = load_model('model.h5')
test_image =image.load_img('state-farm-distracted-driver-detection\imgs\img_6.jpg',target_size = (77,77))
plt.imshow(test_image)

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
classes =loaded_model.predict(test_image)

print (classes)