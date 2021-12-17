# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 00:25:14 2021

@author: SALAH ABIDO
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from  skimage import io
import glob
from skimage.feature import hog 
from skimage.transform import resize
from sklearn import svm
from sklearn.model_selection import train_test_split
import time
start_time = time.time()
#img =io.imread('train\cat.3.jpg',1)
#print(np.shape(img))
#io.imshow(img)

path = glob.glob("traint\*.jpg")
cv_img = []
Y=[]
#counter=0
for img in path:
    if int(str(img)[-5:-4])==0:
        n = io.imread(img)
        resizedImg=resize(n,(128,64))
        if str(img)[7:10]=='cat':
            Y.append(0)
        elif str(img)[7:10]=='dog':
            Y.append(1)
        cv_img.append(resizedImg)

Y=np.reshape(Y,(len(Y),1))

feature=[]
for i in cv_img:
    fd, hog_image = hog(resizedImg, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True)
    feature.append(fd)
X_train, X_test, y_train, y_test = train_test_split(feature, Y,test_size=0.3 ,shuffle=True)

titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']
# we create an instance of SVM and fit out data.
C = 0.1  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(X_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)

for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    predictions = clf.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print("traing accuracyis %s  for{} ".format(titles[i])%(clf.score(X_train,y_train)))
    print("testing accuracyis %s  for{} ".format(titles[i])%(accuracy))
end_time = time.time()
print(end_time - start_time)
#print(lin_svc.score(X_train,y_train))



