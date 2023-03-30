from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import tritonclient.grpc as grpcclient
import numpy as np
import tritonclient.grpc.model_config_pb2 as mc


model_server="localhost:8001"
model_name = "kmeans"
model_version = "1"

try:
	triton_client = grpcclient.InferenceServerClient(url=model_server, verbose=True)
except Exception as e:
	print("channel creation failed: " + str(e))
	sys.exit(1)

# Infer
inputs = []
outputs = []

# Load the diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Split the dataset into training and testing sets
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


input_data = X_test[1].astype(np.float32)
input_data = np.expand_dims(input_data, axis=0)

# Initialize the data
inputs.append(grpcclient.InferInput('float_input', [1, 10], "FP32"))
inputs[0].set_data_from_numpy(input_data)

outputs.append(grpcclient.InferRequestedOutput('label'))

# Test with outputs
results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
# print("response:\n", results.get_response())
statistics = triton_client.get_inference_statistics(model_name=model_name)
# print("statistics:\n", statistics)
if len(statistics.model_stats) != 1:
    print("FAILED: Inference Statistics")
    sys.exit(1)


# Get the output arrays from the results
label_data = results.as_numpy('label')

print("label_data : ", label_data)
