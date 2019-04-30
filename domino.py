
import time
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing 
from sklearn.decomposition import PCA
import numpy as np
import os # Working with files and folders
from PIL import Image # Image processing
from PIL import ImageFilter

### 
### Data can be downloaded from https://www.dropbox.com/sh/s5f38k4l2on5mba/AACNQgXuw1edwEb6oO1w3CfOa?dl=0
### 


start = time.time()
rootdir = os.getcwd()

image_file = 'images.npy'
key_file = 'keys.npy'

def predict_me(image_file_name, scaler, pca):
  pm = Image.open(image_file_name)
  pm = pm.resize([66,66])
  a = np.array(pm.convert('L')).reshape(1,-1)
  #a = np.array(pm.resize([66,66]).convert('L')).reshape(1,-1)) # array 66x66
  a = scaler.transform(a)
  a = pca.transform(a)
  return classifier.predict(a)
  
def crop_image(im, sq_size):
  new_width = sq_size
  new_height = sq_size
  width, height = im.size   # Get dimensions 
  left = (width - new_width)/2
  top = (height - new_height)/2
  right = (width + new_width)/2
  bottom = (height + new_height)/2
  imc = im.crop((left, top, right, bottom))
  return imc 
  
#def filter_image(im):
  # All filter makes it worse
  #imf = im.filter(ImageFilter.EMBOSS)
  #return imf
  
def provide_altered_images(im):
  im_list = [im]
  im_list.append(im.rotate(90))
  im_list.append(im.rotate(180))
  im_list.append(im.rotate(270))
  return im_list

if (os.path.exists(image_file) and os.path.exists(key_file)):
  print("Loading existing numpy's")
  pixel_arr = np.load(image_file)
  key = np.load(key_file)
else:
  print("Creating new numpy's")  
  key_array = []
  pixel_arr = np.empty((0,66*66), "uint8")

  for subdir, dirs, files in os.walk('data'):
    dir_name = subdir.split("/")[-1]    
    if "x" in dir_name:
      for file in files:
        if ".DS_Store" not in file:
          im = Image.open(os.path.join(subdir, file))
          if im.size == (100,100):  
            use_im = crop_image(im,66) # Most images are shot from too far away. This removes portions of it.
            #use_im = filter_image(use_im) # Filters image, but does no good at all
            im_list = provide_altered_images(use_im) # Create extra data with 3 rotated images of every image
            for alt_im in im_list:
              key_array.append(dir_name)  # Each image here is still the same as directory name
              numpied_image = np.array(alt_im.convert('L')).reshape(1,-1) # Converts to grayscale
              #Image.fromarray(np.reshape(numpied_image,(-1,100)), 'L').show()
              pixel_arr = np.append(pixel_arr, numpied_image, axis=0)
          im.close()

  key = np.array(key_array)
  np.save(image_file, pixel_arr)
  np.save(key_file, key)



# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001, C=10, kernel='rbf', class_weight='balanced') # gamma and C from tests
#le = preprocessing.LabelEncoder()
#le.fit(key)
#transformed_key = le.transform(key)
transformed_key = key


X_train, X_test, y_train, y_test = train_test_split(pixel_arr, transformed_key, test_size=0.1,random_state=7)

#scaler = preprocessing.StandardScaler()

pca = PCA(n_components=500, svd_solver='randomized', whiten=True)
# Fit on training set only.
#scaler.fit(X_train)
pca.fit(X_train)
    
# Apply transform to both the training set and the test set.
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)
    
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
    
    
print ("Fit classifier")
classifier = classifier.fit(X_train_pca, y_train)
print ("Score = " + str(classifier.score(X_test_pca, y_test)))
    
# Now predict the value of the domino on the test data:
expected = y_test
    
print ("Predicting")
predicted = classifier.predict(X_test_pca)
    
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
#print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted, labels  =list(set(key))))
end = time.time()
print(end - start)






