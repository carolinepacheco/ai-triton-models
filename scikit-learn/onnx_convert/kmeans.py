#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:59:42 2023

@author: carolinepacheco
"""

from sklearn.datasets import load_diabetes
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load the diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the number of clusters for K-Means algorithm
n_clusters = 3

# Create and train the K-Means clustering model
kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_model.fit(X_train)

# Convert the trained model to ONNX format
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
# Set target_opset to 12 or lower when converting K-Means models
onnx_model = convert_sklearn(kmeans_model, initial_types=initial_type, target_opset=12)

# Save the ONNX model to a file
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())








# check if the onnx  model is valud
# import onnx

# model = onnx.load("model.onnx")
# onnx.checker.check_model(model)
# print("The model is valid.")


# To rcode on my pc
# ----------------------------------------------------------------------------------------
# Create a new virtual environment and activate it:
# python -m venv myenv2
# source myenv2/bin/activate

# Install the required packages with specific versions:
# pip install numpy==1.22.4
# pip install scipy==1.7.3
# pip install scikit-learn==0.24.2
# pip install skl2onnx==1.9.2
# pip install onnxruntime==1.10.0
# pip install protobuf==3.20.2