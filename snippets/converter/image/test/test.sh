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

#!/bin/bash
# Fail on any error.
set -e
# Display commands being run.
set -x
rm -rf out/
mkdir -p out/
# Test keep warm
echo "Test keep warn"
response=$(curl -X GET http://$HOST:$PORT/keep_warm)
if [ '"ok"' == "${response}" ]; then
  echo "Keep warm replied"
else 
  echo "Keep warm didnt reply ok"
  exit 1
fi;
  
# Test image keras
echo "Test image keras conversion"
time curl -X POST \
  http://$HOST:$PORT/convert/image/keras \
  --silent \
  -H 'cache-control: no-cache' \
  -H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  -F model=@./image-model.zip > out/model.h5
file="out/model.h5"
minimumsize=2400000
actualsize=$(wc -c <"$file")
if [ $actualsize -ge $minimumsize ]; then
    echo size is over $minimumsize bytes
else
    echo size is under $minimumsize bytes
    exit 1 
fi
# Test image savedmodel
echo "Test image savedmodel conversion"
time curl -X POST \
  http://$HOST:$PORT/convert/image/savedmodel \
  --silent \
  -H 'cache-control: no-cache' \
  -H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  -F model=@./image-model.zip > out/model.savedmodel
file="out/model.savedmodel"
minimumsize=2400000
actualsize=$(wc -c <"$file")
if [ $actualsize -ge $minimumsize ]; then
    echo size is over $minimumsize bytes
else
    echo size is under $minimumsize bytes
    exit 1 
fi
# Test image edgetpu
echo "Test image edgetpu conversion"
time curl -X POST \
  http://$HOST:$PORT/convert/image/edgetpu \
  --silent \
  -H 'cache-control: no-cache' \
  -H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  -F model=@./image-model.zip \
  -F dataset=@./image-model-data.zip > out/model.tflite
file="out/model.tflite"
minimumsize=900000
actualsize=$(wc -c <"$file")
if [ $actualsize -ge $minimumsize ]; then
    echo size $actualsize is over $minimumsize bytes
else
    echo size $actualsize is under $minimumsize bytes
    exit 1 
fi
  
# Test image tflite
echo "Test image tflite conversion"
time curl -X POST \
  http://$HOST:$PORT/convert/image/tflite \
  --silent \
  -H 'cache-control: no-cache' \
  -H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  -F model=@./image-model.zip \
  -F dataset=@./image-model-data.zip > out/model.zip
unzip out/model.zip -d out
model="out/model_unquant.tflite"
test_response=$(python test-image-tflite.py ${model} ./image-model-data.zip "Class 1/1.png")
if [ '0' != "${test_response}" ]; then
  exit 1
fi;
test_response=$(python test-image-tflite.py ${model} ./image-model-data.zip "Class 2/1.png")
if [ '1' != "${test_response}" ]; then
  exit 1
fi;
