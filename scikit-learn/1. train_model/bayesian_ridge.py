#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:59:42 2023

@author: carolinepacheco
"""

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib

# Set the random seed
random_seed = 42

# Load the California Housing dataset
california = fetch_california_housing()
X, y = california.data, california.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Create and train the Bayesian Ridge Regression model
br_model = BayesianRidge()
br_model.fit(X_train, y_train)

# Save the trained model to disk
joblib.dump(br_model, 'br_model.joblib')
