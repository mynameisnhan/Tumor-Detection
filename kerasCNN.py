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
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import MaxPooling3D
from keras.utils import np_utils
from keras import backend as K
from Data import DataHandler
from keras.models import model_from_json
from keras.layers.convolutional import UpSampling3D
from keras.layers.convolutional import Conv2DTranspose
# K.set_image_dim_ordering('th')

def saveModel(mod, targetDir):
    #serialize model to JSON
    model_json = mod.to_json()
    with open("{}model.json".format(targetDir), "w") as json_file:
        json_file.write(model_json)
    #serialize weights to HDF5
    mod.save_weights("{}model.h5".format(targetDir))
    print("Model saved to {}".format(targetDir))

def loadModel(sourceDir):
    json_file = open("{}model.json".format(sourceDir))
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("{}model.h5".format(sourceDir))
    print("Loaded model from {}".format(sourceDir))
    return loaded_model



handler = DataHandler()
#Loads in the datasets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(type(X_train))
xs, ys = handler.load_samples("samples/subset0/", (1, 10,16,16))
X_train = xs[0:-20000]
y_train = ys[0:-20000]
X_test = xs[-20000:]
y_test = ys[-20000:]
print(X_train.shape)
print(y_train.shape)

#Has constant random seed
seed = 7
numpy.random.seed(seed)

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')



# one hot encode outputs
#Turns each value into a 1 at the correct position of a vector
#EX 6 becomes 1 X 10 Vector --> [0,0,0,0,0,1,0,0,0,0]
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)
num_classes = y_test.shape[1]



# Create the model
model = Sequential()

#Runs 32 (3,3) Kernels (features) over the image which is 3 channels (RGB) and (32 X 32)
#Run activation maps through relu to return 0 for negative values
#Constrains the Kernel to be size 3 at most?
#The output now has 32 activation maps, each for a different feature
model.add(Conv3D(64, (3, 5, 5), input_shape=(1, 10, 16, 16), padding='valid', activation='relu'))
#Sets .2 of values to 0 to avoid overfitting
model.add(Dropout(0.2))
#Runs 32 (3,3) Kernels (larger features) over the 32 activation maps that we have so far. Each kernel has a different value for
#Each activation map, and per kernel the activation map values are summed
#Output is ran through relu to have values of 0 when negative
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='valid'))
model.add(Conv3D(64, (1, 3, 3), activation='relu', padding='valid'))
#Flattens out the activation maps into a 1 dimensional array of values
#In this case we would end up with 32 1-D arrays of values from the 32 activation maps
# Outputs 512 numbers
#You have 512 neurons as output each are one number, beacuse each original neuron has a weight for each value in the vecotr
#You sum each vector value by each weigth and add a bias for all 32 vectors per neuron
model.add(Flatten()) #You flatten so that you can eventually get 1 array, you do not want a ton of activation maps
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.5))
#Takes in 512 numbers, outputs array of 10 numbers
#Each 512 numbers goes into one of the 10 neurons, with weights, summed up
#Value outputted for each class value in the array
#Softmax is used to activate each value between 0 and 1
model.add(Dense(num_classes, activation='softmax'))



#Want to minimize log softmax loss after this.
#Softmax activation is used to give

# Compile model
epochs = 15 # Runs over whole dataset 25 times
lrate = 0.01 #Trains weights with a learning rate of 0.01
decay = lrate/epochs #Decays the influence of the weights over time as you have more epochs ?avoid overfitting?
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False) #Uses SGD with the learning rate and the decay to update weights when necessary
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


#Trains the model you made on the X and y training data set
#Uses X_test and y_test to check accuracy while it is training
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
# Final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))

#saveModel(model, "model/")
loadedModel = loadModel("model/")
print(loadedModel.summary())

weights = loadedModel.get_layer(name="conv3d_3").get_weights()
print(weights[0].shape)

#create fully convolutional network
FCN = Sequential()
FCN.add(Conv3D(64, (3, 5, 5), input_shape=(1, None, None, None), padding='valid', activation='relu'))
FCN.add(loadedModel.get_layer(name="max_pooling3d_1"))
FCN.add(loadedModel.get_layer(name="conv3d_2"))
FCN.add(loadedModel.get_layer(name="conv3d_3"))
FCN.add(Conv3D(150, (2, 2, 2), activation='relu', padding='valid'))
FCN.add(Conv3D(2, (1, 1, 1), activation='relu', padding='valid'))
FCN.compile(loss='mean_squared_error', optimizer='sgd')
print(FCN.summary())
print(FCN.get_layer(name="conv3d_4").get_weights()[0].shape)
print(FCN.get_layer(name="conv3d_5").get_weights()[0].shape)
w1 = loadedModel.get_layer(name="dense_1").get_weights()
w1[0].resize(2, 2, 2, 64, 150)
w2 = loadedModel.get_layer(name="dense_2").get_weights()
w2[0].resize(1, 1, 1, 150, 2)
w3 = loadedModel.get_layer(name="conv3d_1").get_weights()
FCN.get_layer(name="conv3d_4").set_weights(w3)
FCN.get_layer(name="conv3d_5").set_weights(w1)
FCN.get_layer(name="conv3d_6").set_weights(w2)

#do inference on a pice of img
img, _, _ = handler.load_itk_image("data/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd")
img = img[40:50, 240:256, 240:272]
y = FCN.predict(img.reshape(1, 1,img.shape[0],img.shape[1],img.shape[2]), batch_size=1)
print(y.shape)
print(y)
