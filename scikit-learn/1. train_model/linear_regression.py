#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:59:42 2023

@author: carolinepacheco
"""

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Load the California Housing dataset
california = fetch_california_housing()
X, y = california.data, california.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(lr_model, 'linear_regression.pkl')

