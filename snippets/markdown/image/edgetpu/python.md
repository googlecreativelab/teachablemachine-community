To use with the [Coral Edge TPU](https://coral.ai/):

**1.** Install the [Edge TPU Runtime library](https://coral.ai/software/#edgetpu-runtime).

**2.** Install the PyCoral API and other pip packages like so:

```bash
python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0 Pillow opencv-python opencv-contrib-python
```

**3.** Download your model from Teachable Machine.

**4.** Use this code snippet to run this model on the Edge TPU (remember to edit `modelPath` and `labelPath`):

```python
import re
import os
import cv2
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters import classify


# the TFLite converted to be used with edgetpu
modelPath = '<PATH_TO_MODEL>'

# The path to labels.txt that was downloaded with your model
labelPath = '<PATH_TO_LABELS>'

# This function takes in a TFLite Interptere and Image, and returns classifications
def classifyImage(interpreter, image):
    size = common.input_size(interpreter)
    common.set_input(interpreter, cv2.resize(image, size, fx=0, fy=0,
                                             interpolation=cv2.INTER_CUBIC))
    interpreter.invoke()
    return classify.get_classes(interpreter)

def main():
    # Load your model onto the TF Lite Interpreter
    interpreter = make_interpreter(modelPath)
    interpreter.allocate_tensors()
    labels = read_label_file(labelPath)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip image so it matches the training input
        frame = cv2.flip(frame, 1)

        # Classify and display image
        results = classifyImage(interpreter, frame)
        cv2.imshow('frame', frame)
        print(f'Label: {labels[results[0].id]}, Score: {results[0].score}')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

Learn more about how to use the code snippet on [github](https://github.com/google-coral/examples-camera)
