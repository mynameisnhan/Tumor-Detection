from matplotlib import pyplot
from scipy.misc import toimage
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

#Loads in the datasets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#Show some of the images
for i in range(0,9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(toimage(X_train[i]))

#Shows the plot of the images
pyplot.show()

#Has constant random seed

seed = 7
numpy.random.seed(seed)

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0



# one hot encode outputs
#Turns each value into a 1 at the correct position of a vector
#EX 6 becomes 1 X 10 Vector --> [0,0,0,0,0,1,0,0,0,0]
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]



# Create the model
model = Sequential()

#Runs 32 (3,3) Kernels (features) over the image which is 3 channels (RGB) and (32 X 32)
#Run activation maps through relu to return 0 for negative values
#Constrains the Kernel to be size 3 at most?
#The output now has 32 activation maps, each for a different feature
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
#Sets .2 of values to 0 to avoid overfitting
model.add(Dropout(0.2))
#Runs 32 (3,3) Kernels (larger features) over the 32 activation maps that we have so far. Each kernel has a different value for
#Each activation map, and per kernel the activation map values are summed
#Output is ran through relu to have values of 0 when negative
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
#1/4 the current output by polling with a (2,2) Kernel to sample only the largest value in each 2X2 (or 4 pixel) block
model.add(MaxPooling2D(pool_size=(2, 2)))
#Flattens out the activation maps into a 1 dimensional array of values
#In this case we would end up with 32 1-D arrays of values from the 32 activation maps
# Outputs 512 numbers
#You have 512 neurons as output each are one number, beacuse each original neuron has a weight for each value in the vecotr
#You sum each vector value by each weigth and add a bias for all 32 vectors per neuron
model.add(Flatten()) #You flatten so that you can eventually get 1 array, you do not want a ton of activation maps
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
#Takes in 512 numbers, outputs array of 10 numbers
#Each 512 numbers goes into one of the 10 neurons, with weights, summed up
#Value outputted for each class value in the array
#Softmax is used to activate each value between 0 and 1
model.add(Dense(num_classes, activation='softmax'))



#Want to minimize log softmax loss after this.
#Softmax activation is used to give

# Compile model
epochs = 25 # Runs over whole dataset 25 times
lrate = 0.01 #Trains weights with a learning rate of 0.01
decay = lrate/epochs #Decays the influence of the weights over time as you have more epochs ?avoid overfitting?
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False) #Uses SGD with the learning rate and the decay to update weights when necessary
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


#Trains the model you made on the X and y training data set
#Uses X_test and y_test to check accuracy while it is training
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
