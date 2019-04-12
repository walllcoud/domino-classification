from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (10, 10), input_shape = (100, 100, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 7, activation = 'sigmoid')) # 7 units equals amount of output categories

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',
target_size = (100, 100),
batch_size = 21,
class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('dataset/test_set',
target_size = (100, 100),
batch_size = 21,
class_mode = 'categorical')
classifier.fit_generator(training_set,
steps_per_epoch = 8,
epochs = 100,
validation_data = test_set,
validation_steps = 3)
classifier.summary()

# serialize weights to HDF5
classifier.save_weights("dominoweights.h5")
print("Saved model to disk")

# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image

path = 'dataset/prediction_images/' # Folder with my images
for filename in os.listdir(path):
  if "jpg" in filename:
    test_image = image.load_img(path + filename, target_size = (100, 100))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    print result
    training_set.class_indices
    folder = training_set.class_indices.keys()[(result[0].argmax())] # Get the index of the highest predicted value
    if folder == '1':
      prediction = '1x3'
    elif folder == '2':
      prediction = '1x8'
    elif folder == '3':
      prediction = 'Baby'
    elif folder == '4':
      prediction = '5x7'
    elif folder == '5':
      prediction = 'Upside down'
    elif folder == '6':
      prediction = '2x3'   
    elif folder == '7':
      prediction = '0x0'
    else:
      prediction = 'Unknown'
    print "Prediction: " + filename + " seems to be " + prediction
  else:
    print "DSSTORE"
  print "\n"

'''''
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
'''


