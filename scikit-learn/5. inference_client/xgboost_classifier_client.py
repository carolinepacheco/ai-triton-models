from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tritonclient.grpc as grpcclient
import numpy as np
import tritonclient.grpc.model_config_pb2 as mc

from tritonclient import utils

random_seed = 42

model_server="localhost:8001"

try:
    triton_client = grpcclient.InferenceServerClient(url=model_server, verbose=True)
except Exception as e:
    print("channel creation failed: " + str(e))
    sys.exit(1)

model_name = "xgboost_classifier"
model_version = "1"

# Infer
inputs = []
outputs = []
inputs.append(grpcclient.InferInput('input__0', [1, 4], "FP32"))
iris = load_iris()
X, y = iris.data, iris.target
_, X_test, _, y_test = train_test_split(X, y, random_state=random_seed)
input_data = X_test[1].astype(np.float32)
input_data = np.expand_dims(input_data, axis=0)

# Initialize the data
inputs[0].set_data_from_numpy(input_data)

outputs.append(grpcclient.InferRequestedOutput('output__0'))

# Test with outputs
results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
# print("response:\n", results.get_response())
statistics = triton_client.get_inference_statistics(model_name=model_name)
# print("statistics:\n", statistics)
if len(statistics.model_stats) != 1:
    print("FAILED: Inference Statistics")
    sys.exit(1)

# Get the output arrays from the results
output0_data = results.as_numpy('output__0')

print("output0_data : ", output0_data)