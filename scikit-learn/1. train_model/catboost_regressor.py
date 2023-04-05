#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:59:42 2023

@author: carolinepacheco
"""

from catboost import CatBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import joblib

# Set the random seed
random_seed = 42

# Load the California Housing dataset
california = fetch_california_housing()
X, y = california.data, california.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Create and train the CatBoost regressor
cat_boost_model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_seed=random_seed, verbose=0)
cat_boost_model.fit(X_train, y_train)

# Save the trained model to disk
joblib.dump(cat_boost_model, 'cat_boost_model.joblib')



# from catboost import CatBoostRegressor
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split

# # Load the California Housing dataset
# california = fetch_california_housing()
# X, y = california.data, california.target

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train the CatBoost regressor
# cat_boost_model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_seed=42, verbose=0)
# cat_boost_model.fit(X_train, y_train)

# # Save the CatBoost model to ONNX format
# cat_boost_model.save_model("model.onnx", format="onnx")