#By Hoang Huu Viet
import numpy as np
import glob
from PIL import Image
from keras import models
from keras import layers

##read the train images
train_dir = 'ifd-train/'
class_num = 61
train_images = []
train_labels = []
for i in range(1,class_num+1):
    filenames = train_dir  + str(i) + '-*.jpg'
    #print(filenames)
    for filename in glob.glob(filenames):
        #print(filename)
        train_labels.append(i-1)
        train_image = Image.open(filename)
        train_image = np.array(train_image)
        train_images.append(train_image)

train_images = np.array(train_images)

##read the test images
test_dir = 'ifd-test/'
test_images = []
test_labels = []
for i in range(1,class_num+1):
    filenames = test_dir  + str(i) + '-*.jpg'
    for filename in glob.glob(filenames):
        #print(filename)
        test_labels.append(i-1)
        test_image = Image.open(filename)
        test_image = np.array(test_image)
        test_images.append(test_image)

test_images = np.array(test_images)

# print(train_images.shape[1])
# print(train_images.shape[2])
#
# print(test_images.shape)
print(train_labels)
print(test_labels)
y = test_labels
#----------------------------------------------------------

network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape = (train_images.shape[1] * train_images.shape[2],)))
network.add(layers.Dense(class_num, activation = 'softmax'))
network.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_images = train_images.reshape((train_images.shape[0], train_images.shape[1] * train_images.shape[2]))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], test_images.shape[1]*test_images.shape[2]))
test_images = test_images.astype('float32') / 255

print(train_images.shape)
print(test_images.shape)

from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#
print(train_labels.shape)
print(test_labels.shape)

network.fit(train_images, train_labels, epochs = 50)

#test_loss, test_acc = network.evaluate(test_images, test_labels)
test_loss, test_acc = network.evaluate(train_images, train_labels)
print('test_acc:', test_acc)


x = network.predict_classes(test_images[0:20,:])
x = np.array(x)
print(x.shape)
print(x)
print(y)