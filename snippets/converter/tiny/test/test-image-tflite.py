import numpy as np
import tensorflow as tf
import sys
import zipfile
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tempfile
import shutil
import os


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=sys.argv[1])
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Load image
with zipfile.ZipFile(sys.argv[2], 'r') as zip_ref:
    dirpath = tempfile.mkdtemp()
    zip_ref.extractall(dirpath)

    img = load_img(os.path.join(dirpath, sys.argv[3]))
    img_array = img_to_array(img)
    img_array = (img_array.astype(np.float32) / 127.0) - 1

    # Set the tensor
    interpreter.set_tensor(
        input_details[0]['index'], img_array.reshape((1, 224, 224, 3)))
    # Run the model
    interpreter.invoke()

    shutil.rmtree(dirpath)

output_data = interpreter.get_tensor(output_details[0]['index'])
print(np.argmax(output_data))
