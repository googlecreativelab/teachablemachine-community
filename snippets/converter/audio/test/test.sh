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