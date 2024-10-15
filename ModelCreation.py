import os
import numpy as np
import pickle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

categories = ['Organic', 'Recycle']

# Preparing Train Data
input_dir = "./RawData/DATASET/TRAIN"
X_train = []
Y_train = []
for index, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        X_train.append(img.flatten())
        Y_train.append(index)
        print(index,end="")

# X_train, _, Y_train, _ = train_test_split(X_train,Y_train,train_size=0.6,shuffle=True)
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
print("trained complete")

# Preparing Test Data
input_dir = "./RawData/DATASET/TEST"
X_test = []
Y_test = []
for index, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        X_test.append(img.flatten())
        Y_test.append(index)
        print(index, end="")


X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)
print("tested complete")


classifier = SVC()
parameter = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
grid_search = GridSearchCV(classifier, parameter)
model = grid_search.fit(X_train, Y_train)
print("leaning complete")

best_model = grid_search.best_estimator_
Y_predict = best_model.predict(X_test, Y_test)
score = accuracy_score(Y_predict, Y_test)
print(score, " is Accuracy score")

with open('model.p', 'wb') as f:
    pickle.dump(best_model, f)
    f.close()
