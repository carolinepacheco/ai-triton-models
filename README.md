# Deploying AI Models on Triton Server

Triton Server, formerly known as NVIDIA TensorRT Inference Server, is an open-source inference server designed to deploy machine learning models at scale. It provides a flexible and efficient platform for serving AI models in various deep learning frameworks, such as TensorFlow, PyTorch, ONNX Runtime, and TensorRT. In this tutorial, we will explain how to deploy AI models on Triton Server.

This repository provides resources for deploying AI models on Triton Server. The folders contain trained models, converted ONNX models, packaged models, and inference clients.

## Folders

1. **train_model**: This folder contains the trained scikit-learn models in `.model` format. For example, you can find the AdaBoost Regressor model in the `adaboost_regressor.model` file.

2. **convert2onnx**: This folder contains the converted ONNX models for each algorithm. For example, you can find the AdaBoost Regressor ONNX model in the `adaboost_regressor.onnx` file.

3. **triton_model_packaging**: This folder contains the `triton_model_packaging.py` script, which helps you create organized folders and the `config.pbtxt` file to be ready for deployment on Triton Server.

4. **model_repository**: Unzip model to be served by Triton.

5. **inference_client**: This folder contains the Python client scripts for submitting inference requests to the deployed models on Triton Server. For example, you can find the AdaBoost Regressor inference client in the `adaboost_regressor_client.py` file.

## Deployment

In order to use these resources, follow the deployment steps mentioned in the tutorial. Make sure to replace the corresponding file names and paths with the ones provided in these folders.


## Prerequisites

Before we begin, make sure to install the required packages with specific versions:
 ```
    pip install numpy==1.22.4
    pip install scipy==1.7.3
    pip install scikit-learn==0.24.2
    pip install skl2onnx==1.9.2
    pip install onnxruntime==1.10.0
    pip install protobuf==3.20.2
 ```

##  Algorithms (scikit-learn library)

We will cover the following algorithms in this tutorial:

| Algorithm                              | 
|----------------------------------------|
| Isolation Forest                       | 
| CatBoost Regressor                     |
| AdaBoost Regressor                     | 
| Bayesian Ridge                         | 
| Gaussian Naive Bayes                   | 
| Gradient Boosting Regressor            | 
| KMeans                                 |
| Linear Regression                      | 
| Logistic Regression                    | 
| One-Class SVM            				          |
| Random Forest Regressor   			          |
| Support Vector Machines                | 
| Support Vector Regression              | 

# Deploying Models on Triton Server

1. **Prepare the model**: Convert your scikit-learn model to the ONNX format. You can use the codes available in `` `code folder` `` to generate the Onnx model for each algorithm.. Save the ONNX model to a file.

2. **Install Triton Server**: Follow the [Triton Server installation guide](https://github.com/triton-inference-server/server/blob/main/README.md) to install Triton Server on your machine or a supported platform.

3. **Create a model repository**: Triton Server uses a model repository to manage models. Create a directory that will serve as the model repository:
```
mkdir model_repository
```

4. **Create a model directory**: Inside the model repository, create a directory with the name of your model (e.g., `my_model`). Inside the model directory, create another directory named `1` (it represents the version number):
 ```
mkdir -p model_repository/my_model/1
```

5. **Create a config file**: Inside the model directory, create a configuration file named `config.pbtxt`. Here's an example configuration file:
 ```
name: "my_model"
platform: "onnxruntime_onnx"
max_batch_size: 0
input [
    {
        name: "float_input"
        data_type: TYPE_FP32
        dims: [ -1, 4 ]
    }
]
output [
    {
        name: "output_label"
        data_type: TYPE_INT64
        dims: [ -1 ]
    }
instance_group {
    count: 1
    kind: KIND_CPU
}
]
 ```
Update the name, input, and output fields to match your specific model.

6. **Copy the ONNX modele**: Copy the ONNX model file you generated earlier to the version directory (1)

7. **Submitting Inference Requests**: 

With our models now deployed on a running Triton server, let's test them and submit some inferences. 

```bash
# Step 1: Start the inference container in the same network as the triton service
# In a separate console, launch the image_client example from the NGC Triton SDK container
docker run -it --rm --net=host -v ${PWD}/client:/workspace nvcr.io/nvidia/tritonserver:23.02-py3-sdk
bash

# Step 2: Install the required libraries
pip install -r requirements.txt 

# Step 3: Execute some inferences related to the model of your choice
# for example for xgboost run the following script
python xgboost_inference.py
```
