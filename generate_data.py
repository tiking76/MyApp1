from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes = ['kizunaai', 'miraiakari', 'kaguyaruna']
num_classes = len(classes)
IMAGE_SIZE = 244

# image file
X = []

# correct label
Y = []

for index, classlabel in enumerate(classes):
    photo_dir = './dataset/' + classlabel
    files = glob.glob(photo_dir + ('/*.jpg'or'/*.png'or'/*.jpeg'))
    for i, file in enumerate(files):
        image = Image.open(file)
        # standardize to 'RGB'
        image = image.convert('RGB')
        # to make image file all the same size
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save('./image_files.npy', xy)
