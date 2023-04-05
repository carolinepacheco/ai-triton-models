#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:59:42 2023

@author: carolinepacheco
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import joblib

# Set random seed for reproducibility
np.random.seed(42)

random_seed = 42

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed)
clf = SVC(probability=True, random_state=random_seed)
clf.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(clf, 'svm.pkl')
