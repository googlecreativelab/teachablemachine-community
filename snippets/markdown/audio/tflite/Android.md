## Test your model with our sample app

You can test your TensorFlow Lite sound classification model on Android by following these steps:

1. Download the [sample app](https://github.com/tensorflow/examples/tree/master/lite/examples/sound_classification/android)
 from GitHub.
1. Follow instuctions in sample app README to import the app into Android Studio
1. Make the project by going to Build -> Make Project
1. Extract the ZIP archive that you download from Teachable Machine.
1. Copy the `soundclassifier_with_metadata.tflite` file from the archive to the
 `src/main/assets` folder in the sample app, replacing the demo model there.
1. Open the file `src/main/java/org/tensorflow/lite/examples/soundclassifier/MainActivity.kt`
1. Replace the line

    ```kotlin
    private const val MODEL_FILE = "yamnet.tflite"
    ```

    with

    ```kotlin
    private const val MODEL_FILE = "soundclassifier_with_metadata.tflite"
    ```

1. Build and install the app on device

 *Note: Please use a physical Android device to run the sample app.*

## Integrate your model into your own app

You can use TFLite Task Library - AudioClassifier API to integrate the model into your Android. See the TFLite [documentation](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier) for more details.