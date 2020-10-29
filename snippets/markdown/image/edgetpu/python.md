To Use with [Edge TPU](https://coral.withgoogle.com/):

**1.** Install the edgetpu library following [Coral's official instructions](https://coral.withgoogle.com/docs/edgetpu/api-intro/#install-the-library)

**2.** pip install the following packages like so:

```bash
pip3 install Pillow opencv-python opencv-contrib-python
```

**3.** Download model from TM2

**4.** Use this code snippet to run this model on Edge TPU:

```python
from edgetpu.classification.engine import ClassificationEngine
from PIL import Image
import cv2
import re
import os

# the TFLite converted to be used with edgetpu
modelPath = '<PATH_TO_MODEL>'

# The path to labels.txt that was downloaded with your model
labelPath = '<PATH_TO_LABELS>'

# This function parses the labels.txt and puts it in a python dictionary
def loadLabels(labelPath):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(labelPath, 'r', encoding='utf-8') as labelFile:
        lines = (p.match(line).groups() for line in labelFile.readlines())
        return {int(num): text.strip() for num, text in lines}

# This function takes in a PIL Image and the ClassificationEngine
def classifyImage(image, engine):
    # Classify and ouptut inference
    classifications = engine.ClassifyWithImage(image)
    return classifications

def main():
    # Load your model onto your Coral Edgetpu
    engine = ClassificationEngine(modelPath)
    labels = loadLabels(labelPath)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Format the image into a PIL Image so its compatable with Edge TPU
        cv2_im = frame
        pil_im = Image.fromarray(cv2_im)

        # Resize and flip image so its a square and matches training
        pil_im.resize((224, 224))
        pil_im.transpose(Image.FLIP_LEFT_RIGHT)

        # Classify and display image
        results = classifyImage(pil_im, engine)
        cv2.imshow('frame', cv2_im)
        print(results)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

Learn more about how to use the code snippet on [github](https://github.com/google-coral/examples-camera)
