#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Mar 31 15:29:06 2023
@author: carolinepacheco
"""

import onnx
import shutil
import os
import zipfile

def create_triton_config(model_path, config_path, model_name, max_batch_size=0):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Extract input and output information
    input_tensors = []
    for i in model.graph.input:
        shape = [dim.dim_value if dim.dim_value >= 1 else -1 for dim in i.type.tensor_type.shape.dim][1:]
        input_tensors.append({"name": i.name, "data_type": "TYPE_FP32", "dims": shape})

    output_tensors = []
    for o in model.graph.output:
        shape = [dim.dim_value if dim.dim_value >= 1 else -1 for dim in o.type.tensor_type.shape.dim]
    # Create the Triton configuration
    config = {
        "name": model_name,
        "backend": "onnxruntime",
        "max_batch_size": max_batch_size,
        "input": input_tensors,
        "output": output_tensors,
        "instance_group": [{"count": 1, "kind": "KIND_CPU"}],
    }

    # Save the configuration as a JSON file
    with open(config_path, 'w') as f:
        f.write("name: \"" + config['name'] + "\"\n")
        f.write("backend: \"" + config['backend'] + "\"\n")
        f.write("max_batch_size: " + str(config['max_batch_size']) + "\n")
        f.write("input [\n")
        for input_tensor in config['input']:
            f.write("  {\n")
            f.write("    name: \"" + input_tensor['name'] + "\"\n")
            f.write("    data_type: " + input_tensor['data_type'] + "\n")
            f.write("    dims: [ " + ", ".join([str(dim) for dim in input_tensor['dims']]) + " ]\n")
            f.write("  }\n")
        f.write("]\n")
        f.write("output [\n")
        for output_tensor in config['output']:
            f.write("  {\n")
            f.write("    name: \"" + output_tensor['name'] + "\"\n")
            f.write("    data_type: " + output_tensor['data_type'] + "\n")
            f.write("    dims: [ " + ", ".join([str(dim) for dim in output_tensor['dims']]) + " ]\n")
            f.write("  }\n")
        f.write("]\n")
        f.write("instance_group [\n")
        for instance_group in config['instance_group']:
            f.write("  {\n")
            f.write("    count: " + str(instance_group['count']) + "\n")
            f.write("    kind: " + instance_group['kind'] + "\n")
            f.write("  }\n")
        f.write("]\n")

    print(f"The configuration file has been saved to '{config_path}'")



def list_onnx_files(directory):
    onnx_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".onnx"):
                onnx_files.append(file)

    return onnx_files



def zip_folder(folder_path, output_path):
    # Create a ZipFile object
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Walk through the directory tree and add each file to the zip file
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, arcname=os.path.relpath(file_path, folder_path))
                
            # Add an empty directory entry for each subdirectory
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                zip_file.write(dir_path, arcname=os.path.relpath(dir_path, folder_path))
            
    print(f"The folder '{folder_path}' has been zipped to '{output_path}'")



folder_path = "/Users/carolinepacheco/Desktop/server/test2/scikit-learn/packgaging"

# Change to the parent directory
os.chdir(folder_path)

filenames = list_onnx_files(folder_path)
version = '1'
print(filenames)

for filename in filenames:
    if not filename.startswith("."):  # Exclude hidden files
        foldername = os.path.splitext(filename)[0]  # Get the filename without extension
        if not os.path.exists(foldername):  # Check if folder doesn't exist
            os.makedirs(foldername, exist_ok=True)
            folderdir = os.path.join(foldername, version)
            os.makedirs(folderdir, exist_ok=True)
            shutil.move(filename, folderdir)  # Move the file to the folder

            model_path = os.path.join(folderdir, filename)
            config_path = os.path.join(foldername, "config.pbtxt")
            create_triton_config(model_path, config_path, foldername, max_batch_size=0)

            os.rename(os.path.join(folderdir, filename), os.path.join(folderdir, 'model.onnx'))  # Rename the file
            print(f"The file '{os.path.join(folderdir, filename)}' has been renamed to '{os.path.join(folderdir, 'model.onnx')}'")

            print(f"{foldername} folder created successfully!")
            print(f"{filename} moved to {foldername} successfully!")
            zip_folder(os.path.join(folder_path, foldername), f"{foldername}.zip")
        else:
            print(f"{foldername} folder already exists.")
