```python
import tensorflow.keras
from PIL import Image
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open('test_photo.jpg')


# resizes the image proportionally to 224 pixels tall
scaled_height = 224
scaled_width = int((scaled_height / image.height) * image.width)
image = image.resize((scaled_width, scaled_height))

#turn the image into a numpy array
image_array = np.asarray(image)

# this crops resized image to its center
crop_factor = (image.width - 224) // 2
# if the input image has an odd width, we need to add 1 so we still crop to a 
# width of 224
shift_by_1 = 0 if image.width % 2 == 0 else 1  
image_array = image_array[:, crop_factor + shift_by_1:(image.width - crop_factor)]

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
print(prediction)

