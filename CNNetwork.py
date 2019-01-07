#By Hoang Huu Viet
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import glob
from PIL import Image
from keras import models
from keras import layers

#read the train images
train_dir = 'att-train/'
num_classes = 36
train_images = []
train_labels = []
for i in range(1,num_classes + 1):
    filenames = train_dir  + str(i) + '-*.jpg'
    #print(filenames)
    for filename in glob.glob(filenames):
        #print(filename)
        train_labels.append(i-1)
        train_image = Image.open(filename)
        train_image = np.array(train_image)
        train_images.append(train_image)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

#read the test images
test_dir = 'att-test/'
test_images = []
test_labels = []
for i in range(1,num_classes + 1):
    filenames = test_dir  + str(i) + '-*.jpg'
    for filename in glob.glob(filenames):
        #print(filename)
        test_labels.append(i-1)
        test_image = Image.open(filename)
        test_image = np.array(test_image)
        test_images.append(test_image)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

#create random rows
r = np.arange(0,len(train_labels))
np.random.shuffle(r)
train_images = train_images[r,:];
train_labels = train_labels[r];

# display the first 100 images
# import matplotlib.pyplot as plt
# fig = plt.figure()
# cols = 10
# rows = 10
# for i in range(1,rows*cols+1):
#     image = train_images[i-1]
#     fig.add_subplot(rows,cols,i)
#     plt.axis('off')
#     plt.imshow(image,cmap = 'gray')
# plt.show()

#get the image parameters
num_train_samples = train_images.shape[0]
num_test_samples = test_images.shape[0]
height = train_images.shape[1]
width = train_images.shape[2]

#normalize the data samples
train_images = train_images.reshape((num_train_samples, height, width,1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((num_test_samples, height, width,1))
test_images = test_images.astype('float32') / 255

#encode the targers
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
y = test_labels
test_labels = to_categorical(test_labels)

#create samples for simple hold-out validation
val_train_images = train_images[:50]
val_train_labels = train_labels[:50]
partial_train_images = train_images[50:]
partial_train_labels = train_labels[50:]
#--------------------------------------------------------------------
#create a network model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height,width,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.35))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.35))

# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.35))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#train the network
history = model.fit(partial_train_images, partial_train_labels, epochs = 20, batch_size = 8, validation_data=(val_train_images, val_train_labels),shuffle = True)

#evaluate the trained netwotk
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

import numpy as np
print(np.argmin(history.history['val_loss']))

#output labels
print(y)
x = model.predict_classes(test_images)
print(np.array(x))

#plot training loss and validation loss
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(history.history['loss'],'-b^',label = 'Training loss')
plt.plot(history.history['val_loss'],'-rv',label = 'Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(loc = 'upper right')

#plot training accurancy and validation accurancy
plt.figure(2)
plt.plot(history.history['acc'],'-b>',label = 'Training accurancy')
plt.plot(history.history['val_acc'],'-r<',label = 'Validation accurancy')
plt.ylabel('Accurancy')
plt.xlabel('Epochs')
plt.legend(loc = 'upper left')
plt.show()

