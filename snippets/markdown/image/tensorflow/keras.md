```python
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model

# Load image with right shape to feed into keras model
pic = image.load_img('<image path>',target_size=(224,224))

# Load the model
model = load_model('keras_model.h5')

# Display Image
plt.imshow(testing)

# Convert image to array of required type
testing = image.img_to_array(testing)

# Predict image through model
k = model.predict_classes(testing.reshape(1,28,28,1))

# Print Prediction
print(k)
