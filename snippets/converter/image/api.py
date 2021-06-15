# ==============================================================================
# Copyright 2021 Google LLC All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


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
from tensorflow.keras.preprocessing import image
import tempfile
import shutil
import zipfile
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
labels = []
datapath = ""
instanceReady = True
# load preproc layers
def representative_dataset_gen():
    global labels, datapath
    for label_index in range(len(labels)):
        img_folder_path = datapath + '/' + labels[label_index]
        dirListing = os.listdir(img_folder_path)
        for f in dirListing:
            if not f.startswith('.'):
                img = image.load_img(path=os.path.join(
                    img_folder_path, f), grayscale=False)
                img = ((image.img_to_array(img) / 127.5) - 1.0).astype(np.float32)
                img = img.reshape(1, 224, 224, 3)
                yield [img]
def converterSavedModelTFLite(pathToSavedModel):
    global labels
    converter = tf.lite.TFLiteConverter.from_saved_model(pathToSavedModel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.representative_dataset = representative_dataset_gen
    converter.allow_custom_ops = True
    converter.change_concat_input_ranges = True
    return converter.convert()
def convertSavedModelTFLiteUnQuantized(pathToSavedModel):
    global labels
    converter = tf.lite.TFLiteConverter.from_saved_model(pathToSavedModel)
    return converter.convert()
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
    
    if ((format == 'edgetpu' or format == 'tflite_quantized')and dataset == None):
        return {'No representative dataset supplied'}
    model_dir = tempfile.mkdtemp()
    print("### Created "+model_dir)
    data_dir = tempfile.mkdtemp() 
    unzipFile(model, model_dir)
    
    background_tasks.add_task(cleanup_files, model_dir, data_dir)
    
    with open(model_dir + '/metadata.json') as json_file:
        data = json.load(json_file)
    
    labels = data['labels']
    
    print("Generating lables.txt")
    with open(model_dir + '/labels.txt', 'w') as f:
        for idx, label in enumerate(labels):
            f.write("{} {}\n".format(idx, label))
    print('Labels:'+', '.join(labels), flush=True)
    
    print('converting model to keras', flush=True)
    os.system('tensorflowjs_converter --input_format tfjs_layers_model --output_format keras "' +
                model_dir + '/model.json" ' + model_dir + '/keras_model.h5')
    if format == 'keras':
        return returnFile('keras_model.h5', model_dir, data_dir)
    # Generate savedmodel
    print('converting model to saved model', flush=True)
    model = tf.keras.models.load_model(model_dir + '/keras_model.h5')
    model.save(model_dir + '/model.savedmodel')
    
    if format == 'savedmodel':
        return returnFile('model.savedmodel', model_dir, data_dir, True)
    
    if format == 'tflite':
        # Generate tflite unquantized
        tflite_unquant_model = convertSavedModelTFLiteUnQuantized(
            model_dir + '/model.savedmodel')
        open(model_dir + '/model_unquant.tflite', 'wb').write(tflite_unquant_model)
        return returnFile('model_unquant.tflite', model_dir, data_dir)
    
    # Generate tflite
    unzipFile(dataset, data_dir)
    datapath = data_dir
    print('convert model to tflite', flush=True)
    tflite_quant_model = converterSavedModelTFLite(
        model_dir + '/model.savedmodel')
    open(model_dir + '/model.tflite', 'wb').write(tflite_quant_model)
    
    if format == 'tflite_quantized':
        return returnFile('model.tflite', model_dir, data_dir)
    # Generate edgetpu model
    print('compile model for edgetpu', flush=True)
    os.system('edgetpu_compiler -s ' + model_dir +
                '/model.tflite -o ' + model_dir)
    
    if format == 'edgetpu':
        return returnFile('model_edgetpu.tflite', model_dir, data_dir)
    else:
        return {'invalid format': format}