#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:59:42 2023

@author: carolinepacheco
"""

from sklearn.datasets import load_diabetes
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

# Set random seed for reproducibility
np.random.seed(42)

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

# Save the trained model to a file
joblib.dump(kmeans_model, 'kmeans.pkl')


