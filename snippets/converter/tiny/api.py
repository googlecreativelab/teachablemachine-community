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
import PIL
import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import tempfile
import shutil
import zipfile
from string import Template

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
labels = []
datapath = ""
instanceReady =  True

def representative_dataset_gen():
    global labels, datapath
    
    for label_index in range(len(labels)):
        img_folder_path = datapath + '/' + labels[label_index]
        dirListing = os.listdir(img_folder_path)

    for f in dirListing:
        if not f.startswith('.'):
            img = PIL.Image.open(os.path.join(img_folder_path, f))
            img = img.resize((96, 96))
            img = img.convert('L')
            array = np.array(img)
            array = np.expand_dims(array, axis=2)
            array = np.expand_dims(array, axis=0)
            array = ((array / 127.5) - 1.0).astype(np.float32)
            yield ([array])

def unzipFile(file, job_dir):
    print('unzipping!')
    with zipfile.ZipFile(file.file, 'r') as model_data:
        model_data.extractall(job_dir)
    return True

def returnFolder(folderName, model_dir):
    global instanceReady
    instanceReady = True
    shutil.make_archive(model_dir + '/retZip', 'zip', model_dir + '/tm_template_script')
    return FileResponse(model_dir + '/retZip.zip', media_type='application/octet-stream', filename='arduino_sketch.zip')

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
def format_labels():
    retStr = ''
    for i in range(len(labels)):
        label = labels[i]
        retStr += '"{}",'.format(label)

    return retStr
def format_arduino_sketch(model_dir):
    #read model flatbuffer
    with open(model_dir + '/output_model.cc', 'r') as f:
        data = f.read()
        model_buffer = data[data.find('{')+1 : data.find('}')]
        search_string = 'len = '
        model_buffer_len = data[data.find(search_string) + len(search_string) : data.rfind(';')]
        print('parsed ' + model_buffer_len + ' model flatbuffer bytes')

    #write model buffer into arduino file
    with open(model_dir + '/tm_template_script/person_detect_model_data.cpp', 'r+') as f:
        model_template = Template(f.read())
        new_file = model_template.safe_substitute({ 'model_buf': model_buffer, 'model_buf_len': model_buffer_len })
        f.seek(0)
        f.write(new_file)
        f.truncate()
    #write model settings
    with open(model_dir + '/tm_template_script/model_settings.h', 'r+') as f:
        # data = f.read()
        # print(data)
        settings_template = Template(f.read())
        # print('setting num clases to', len(labels))
        new_file = settings_template.safe_substitute({ 'numClasses': len(labels)})
        f.seek(0)
        f.write(new_file)
        f.truncate()

    with open(model_dir + '/tm_template_script/model_settings.cpp', 'r+') as f:
        class_labels_template = Template(f.read())
        # print('new labels', format_labels())
        new_file = class_labels_template.safe_substitute({ 'labels': format_labels()})
        f.seek(0)
        f.write(new_file)
        f.truncate()


@app.get("/keep_warm")
async def keep_warm():
    return "ok"

@app.post("/convert/{type}/{format}")
async def create_upload_file(type: str, format: str, background_tasks: BackgroundTasks,  model: UploadFile = File(...), dataset: UploadFile = File(default=None)):
    
    print("uploading", flush=True)
    global labels, datapath,instanceReady
    instanceReady = False
    
    if (type != 'tiny_image' or (format == "tinyml" and dataset == None)):
        raise HTTPException(status_code=403, detail="bad request format")
    
    model_dir = tempfile.mkdtemp()
    print("### Created "+model_dir)
    data_dir = tempfile.mkdtemp()
    unzipFile(model, model_dir)
    background_tasks.add_task(cleanup_files, model_dir, data_dir)
    
    os.system('cp -r tm_template_script ' + model_dir)

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
        return returnFile('keras_model.h5', model_dir, data_dir); 
    
    converter = tf.lite.TFLiteConverter.from_keras_model_file(model_dir + '/keras_model.h5')
    converter.optimizations=[tf.lite.Optimize.DEFAULT]
    
    if format == 'tflite':
        tf_quant_model = converter.convert()
        open(model_dir + '/vww_96_grayscale_quantized.tflite', 'wb').write(tf_quant_model)
        return returnFile('vww_96_grayscale_quantized.tflite', model_dir, data_dir)
    if format == 'tinyml':
        unzipFile(dataset, data_dir)
        datapath = data_dir
        converter.inference_input_type = tf.lite.constants.INT8
        converter.inference_output_type = tf.lite.constants.INT8
        converter.representative_dataset = representative_dataset_gen
        tf_quant_model = converter.convert()
        open(model_dir + '/vww_96_grayscale_quantized.tflite', 'wb').write(tf_quant_model)
        os.system('xxd -i ' + model_dir + '/vww_96_grayscale_quantized.tflite > ' + model_dir + '/output_model.cc' )

        format_arduino_sketch(model_dir)

        return returnFolder('/tm_template_script', model_dir)
