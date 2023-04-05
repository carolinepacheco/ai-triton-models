#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:36:30 2023

@author: carolinepacheco
"""

import joblib
from sklearn.datasets import fetch_california_housing, load_diabetes, load_iris
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def load_dataset(dataset_name):
    if dataset_name == 'california':
        dataset = fetch_california_housing()
    elif dataset_name == 'diabetes':
        dataset = load_diabetes()
    elif dataset_name == 'iris':
        dataset = load_iris()
    else:
        raise ValueError("Invalid dataset name")
    return dataset.data, dataset.target

def prepare_onnx_conversion_params(X, target_opset, model):
    if target_opset in {9, 17}:
        tensor = FloatTensorType([None, 4])
    else:
        tensor = FloatTensorType([None, X.shape[1]])

    if target_opset == 9:
        options = {id(model): {'zipmap': False}}
    else:
        options = None

    return tensor, options

def convert2onnx(model, initial_type, options, target_opset):
    try:
        model.save_model("model.onnx", format="onnx")
        print("CatBoost model saved in ONNX format successfully.")
    except Exception as e:
        print("Error occurred while saving CatBoost model in ONNX format:", e)

    try:
        # Try to convert the trained model to ONNX format
        onnx_model = convert_sklearn(model, initial_types=initial_type, options=options, target_opset=target_opset)
        # Save the ONNX model to a file
        with open("model.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
        print("Model converted to ONNX format and saved successfully.")
    except Exception as e:
        print("Error occurred while converting model to ONNX format:", e)
        

if __name__ == "__main__":
    # Load the dataset
    dataset_name = "california"  # Choose from: 'california', 'diabetes', or 'iris'
    X, y = load_dataset(dataset_name)

    # Load the trained model from disk
    model_path = 'svr.pkl'  # Update this path to the correct model file
    model = joblib.load(model_path)

    # Prepare the ONNX conversion parameters
    target_opset = 12  # Update this value based on your model type
    tensor, options = prepare_onnx_conversion_params(X, target_opset, model)

    input_name = 'float_input'
    initial_type = [(input_name, tensor)]

    # Convert the model to ONNX format
    convert2onnx(model, initial_type, options, target_opset)
