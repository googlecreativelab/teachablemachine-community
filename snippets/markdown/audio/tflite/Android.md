## Test your model with our sample app

You can test your TensorFlow Lite sound classification model on Android by following these steps:

1. Download the [sample app](https://github.com/tensorflow/examples/tree/master/lite/examples/sound_classification/android)
 from GitHub.
1. Copy the `soundclassifier.tflite` file downloaded from Teachable Machine to the
 `src/main/assets` folder in the sample app, replacing the demo model there.

 *Note: Please use a physical Android device to run the sample app.*

## Integrate your model into your own app

You can use TFLite Task Library - AudioClassifier API to integrate the model into your Android. See the TFLite [documentation](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier) for more details.
