from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import os


iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)



parameters = {
   "eta": 0.3,
   "objective": "multi:softprob",  # error evaluation for multiclass tasks
   "num_class": 3,  # number of classes to predic
   "max_depth": 3,  # depth of the trees in the boosting process
}

num_round = 20  # the number of training iterations
clf = xgb.train(parameters, dtrain, num_round)

# Create the model repository directory. The name of this directory is arbitrary.
MODEL_REPO_PATH = 'model_repository'
def serialize_model(model, model_name):
    # The name of the model directory determines the name of the model as reported
    # by Triton
    model_dir = os.path.join(MODEL_REPO_PATH, model_name)
    # We can store multiple versions of the model in the same directory. In our
    # case, we have just one version, so we will add a single directory, named '1'.
    version_dir = os.path.join(model_dir, '1')
    os.makedirs(version_dir, exist_ok=True)
    
    # The default filename for XGBoost models saved in json format is 'xgboost.json'.
    # It is recommended that you use this filename to avoid having to specify a
    # name in the configuration file.
    model_file = os.path.join(version_dir, 'xgboost.json')
    model.save_model(model_file)
    
    return model_dir

model_dir = serialize_model(clf, 'xgboost')

# Maximum size in bytes for input and output arrays. If you are
# using Triton 21.11 or higher, all memory allocations will make
# use of Triton's memory pool, which has a default size of
# 67_108_864 bytes. This can be increased using the
# `--cuda-memory-pool-byte-size` option when the server is
# started
MAX_MEMORY_BYTES = 60_000_000

features = X_test.shape[1]
num_classes = len(np.unique(y_test))
bytes_per_sample = (features + num_classes) * 4
max_batch_size = MAX_MEMORY_BYTES // bytes_per_sample

def generate_config(model_dir, deployment_type='gpu', storage_type='AUTO'):
    if deployment_type.lower() == 'cpu':
        instance_kind = 'KIND_CPU'
    else:
        instance_kind = 'KIND_GPU'

    config_text = f"""backend: "fil"
input [                                 
 {{  
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 1, {features} ]                    
  }} 
]
output [
 {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ {num_classes} ]
  }}
]
instance_group [{{ kind: {instance_kind} }}]
parameters [
  {{
    key: "model_type"
    value: {{ string_value: "xgboost_json" }}
  }},
  {{
    key: "predict_proba"
    value: {{ string_value: "true" }}
  }},
  {{
    key: "output_class"
    value: {{ string_value: "true" }}
  }},
  {{
    key: "threshold"
    value: {{ string_value: "0.5" }}
  }},
  {{
    key: "storage_type"
    value: {{ string_value: "{storage_type}" }}
  }}
]

dynamic_batching {{
  max_queue_delay_microseconds: 100
}}"""
    config_path = os.path.join(model_dir, 'config.pbtxt')
    with open(config_path, 'w') as file_:
        file_.write(config_text)

    return config_path

generate_config(model_dir, deployment_type='cpu')
