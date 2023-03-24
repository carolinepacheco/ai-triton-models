#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:59:42 2023

@author: carolinepacheco
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from sklearn.naive_bayes import GaussianNB

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train a Gaussian Naive Bayes classifier
clf = GaussianNB()
clf.fit(X_train, y_train)
print(clf)

# Specify the input type of the ONNX model
initial_type = [('float_input', FloatTensorType([None, 4]))]

# Set the options for converting the model
options = {id(clf): {'zipmap': False}}

# Convert the model to ONNX format
# Set target_opset to 9 or lower when converting Gaussian Naive Bayes models
onx = convert_sklearn(clf, initial_types=initial_type, options=options, target_opset=9)

# Save the ONNX model to a file
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())