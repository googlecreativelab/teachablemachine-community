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
# echo response
if [ '"ok"' == "${response}" ]; then
  echo "Keep warm replied"
else 
  echo "Keep warm didnt reply ok"
  exit 1
fi;
  
# Test audio keras
echo "Test audio tflite conversion"
time curl -X POST \
  http://$HOST:$PORT/convert/audio/tflite \
  --silent \
  -H 'cache-control: no-cache' \
  -H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  -F model=@./audio-model.zip > out/model.tflite
file="out/model.tflite"
minimumsize=4400000
actualsize=$(wc -c <"$file")
if [ $actualsize -ge $minimumsize ]; then
    echo size is over $minimumsize bytes
else
    echo size is under $minimumsize bytes
    exit 1 
fi
echo "Test sucessful"