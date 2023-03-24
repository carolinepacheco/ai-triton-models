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
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = LogisticRegression(max_iter=500)
clr.fit(X_train, y_train)
print(clr)

initial_type = [('float_input', FloatTensorType([None, 4]))]
options = {id(clr): {'zipmap': False}}

# Set target_opset to 17 or lower when converting the model
onx = convert_sklearn(clr, initial_types=initial_type, options=options, target_opset=17)

with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
    