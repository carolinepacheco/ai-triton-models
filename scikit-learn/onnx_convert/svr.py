#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:59:42 2023

@author: carolinepacheco
"""

from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load the California Housing dataset
california = fetch_california_housing()
X, y = california.data, california.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Support Vector Regression model
svr_model = SVR()
svr_model.fit(X_train, y_train)

# Convert the trained model to ONNX format
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
# Set target_opset to 12 or lower when converting Support Vector Regression models
onnx_model = convert_sklearn(svr_model, initial_types=initial_type, target_opset=12)

# Save the ONNX model to a file
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())