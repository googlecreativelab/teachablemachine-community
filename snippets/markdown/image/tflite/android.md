For this Teachable Machine example, the TFlite model format is being used.

**1.** Download the Tensorflow Examples repository [Github](https://github.com/tensorflow/examples)

(`git clone https://github.com/tensorflow/examples.git`).

**2.** Unpack the _converted_tflite_model.zip_ archive exported from Teachable Machine.

**3.** Copy the _TFlite model_ and the _labels.txt file_ to the asset folder:

`examples/tree/master/lite/codelabs/flower_classification/android/finish/app/src/main/assets`.

**4.** Install and run _Android Studio_.

**5.** Click on "Open an existing Android Studio project".

In the file selector, `choose examples/lite/codelabs/flower_classification/android/finish`.

You will get a "Gradle Sync" popup, the first time you open the project, asking about using gradle wrapper. Click "OK".

**6.** You can [run the App on a virtual device](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/?hl=de#7) for testing.

To build the APK, go to the _Build_ menu -> _Build Bundle(s) / APKs_ -> _Build APKs_.

The menu will be greyed out as long as the project is still being loaded.

A more detailed tutorial can be found here: [Recognize Flowers with TensorFlow Lite on Android Tutorial](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/).
