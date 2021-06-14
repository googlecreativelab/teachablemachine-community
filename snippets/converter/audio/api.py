from starlette.responses import FileResponse
from starlette.requests import Request
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks

from typing import List
import os
import json

import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np

import tempfile
import shutil
import zipfile
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
labels = []
datapath = ""
# load preproc layers
preproc_model_path = 'sc_preproc_model'
preproc_model = tf.keras.models.load_model(preproc_model_path)
input_length = preproc_model.input_shape[-1]
instanceReady = True

def unzipFile(file, job_dir):
    with zipfile.ZipFile(file.file, 'r') as model_data:
        model_data.extractall(job_dir)
    return True
def returnFile(filename, model_dir, data_dir, isSavedModel=False):
    with zipfile.ZipFile(model_dir + '/retZip.zip', 'w') as zip:
        # If the return type is savedmodel, recursively add the files in the savedmodel directory.
        if (isSavedModel):
            for dirname, subdirs, files in os.walk(model_dir + '/' + filename):
                # We use  relpath to remove the /tmp/xxxxx/ from the archive paths. 
                zip.write(dirname, os.path.relpath(dirname, model_dir))
                for filename in files:
                    zip.write(os.path.join(dirname, filename), os.path.relpath(os.path.join(dirname, filename), model_dir))
        else:
            zip.write(model_dir + '/' + filename, filename)
        
        zip.write(model_dir + '/labels.txt', 'labels.txt')
    
    global instanceReady
    instanceReady = True
    
    return FileResponse(model_dir + '/retZip.zip',
                            media_type='application/octet-stream', filename='converted_model.zip')
def cleanup_files(model_dir, data_dir):
    print("### DELETING FILES "+model_dir, flush=True)
    shutil.rmtree(model_dir)
    shutil.rmtree(data_dir)

@app.get("/keep_warm")
async def keep_warm():
    return "ok"

@app.post("/convert/{type}/{format}")
async def create_upload_file(type: str, format: str, background_tasks: BackgroundTasks,  model: UploadFile = File(...), dataset: UploadFile = File(default=None)):
    
    print("uploading", flush=True)
    global labels, datapath, instanceReady
    instanceReady = False
    if (type != 'audio' or format != 'tflite'):
        return {'format not supported'}
    model_dir = tempfile.mkdtemp()
    print("### Created "+model_dir)
    data_dir = tempfile.mkdtemp()
    unzipFile(model, model_dir)
    
    background_tasks.add_task(cleanup_files, model_dir, data_dir)
    
    with open(model_dir + '/metadata.json') as json_file:
        data = json.load(json_file)
    labels = data['wordLabels']
    
    print("Generating lables.txt")
    with open(model_dir + '/labels.txt', 'w') as f:
        for idx, label in enumerate(labels):
            f.write("{} {}\n".format(idx, label))
    print('Labels:'+', '.join(labels), flush=True)

    # specify path to original model and load
    tfjs_model_json_path = model_dir + '/model.json'
    model = tfjs.converters.load_keras_model(tfjs_model_json_path)
    
    # construct the new model by combining preproc and main classifier
    combined_model = tf.keras.Sequential(name='combined_model')
    combined_model.add(preproc_model)
    combined_model.add(model)
    combined_model.build([None, input_length])
    # save the model as a tflite file
    tflite_output_path = model_dir + '/soundclassifier.tflite'
    converter = tf.lite.TFLiteConverter.from_keras_model(combined_model)
    converter.target_spec.supported_ops=[
        tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS 
    ]
    with open(tflite_output_path, 'wb') as f:
        f.write(converter.convert())
    return returnFile('soundclassifier.tflite', model_dir, data_dir, False)