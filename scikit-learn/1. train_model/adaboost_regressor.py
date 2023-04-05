#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:59:42 2023

@author: carolinepacheco
"""

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
import joblib

# Set the random seed
random_seed = 42

# Load the California Housing dataset
california = fetch_california_housing()
X, y = california.data, california.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Create and train the AdaBoost regressor
ada_boost_model = AdaBoostRegressor(n_estimators=100, random_state=random_seed)
ada_boost_model.fit(X_train, y_train)

# Save the trained model to disk
joblib.dump(ada_boost_model, 'ada_boost_model.joblib')

    