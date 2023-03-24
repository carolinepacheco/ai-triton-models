# Deploying AI Models on Triton Server

Triton Server, formerly known as NVIDIA TensorRT Inference Server, is an open-source inference server designed to deploy machine learning models at scale. It provides a flexible and efficient platform for serving AI models in various deep learning frameworks, such as TensorFlow, PyTorch, ONNX Runtime, and TensorRT. In this tutorial, we will explain how to deploy AI models on Triton Server.

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

| Algorithm                              | Dataset                |
|----------------------------------------|------------------------|
| Isolation Forest                       | Diabetes               |
| CatBoost Regressor                     | California Housing     |
| AdaBoost Regressor                     | California Housing     |
| Bayesian Ridge                         | California Housing     |
| Gaussian Naive Bayes                   | Iris                   |
| Gradient Boosting Regressor            | California Housing     |
| KMeans                                 | Diabetes               |
| Linear Regression                      | California Housing     |
| Logistic Regression                    | Iris                   |
| One-Class SVM            				          | Diabetes               |
| Random Forest Regressor   			          | California Housing     |
| Support Vector Machines                | Iris                   |
| Support Vector Regression              | California Housing     |

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
