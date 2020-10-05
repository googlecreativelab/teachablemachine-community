## Test your model with our sample app

You can test your TensorFlow Lite sound classification model on Android by following these steps:

1. Download the [sample app](https://github.com/tensorflow/examples/tree/master/lite/examples/sound_classification/android)
 from GitHub.
2. Extract the ZIP archive that you download from Teachable Machine.
3. Copy the `soundclassifier.tflite` and `labels.txt` files from the archive to the `src/main/assets` folder in the sample app, replacing the demo model there.

## Integrate your model into your own app

If you want to integrate the model into your existing app, follow these steps:

1. Put the `soundclassifier.tflite` and `labels.txt` files into the `assets` folder in your app.
2. Copy the [SoundClassifier.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/sound_classification/android/app/src/main/java/org/tensorflow/lite/examples/soundclassifier/SoundClassifier.kt)
 file to your app. This file contains the source code to use the sound classification model.
3. Initialize a `SoundClassifier` instance from your `Activity` or `Fragment` class.

```kotlin
var soundClassifier: SoundClassifier
soundClassifier = SoundClassifier(context).also {
    it.lifecycleOwner = context // or viewLifecycleOwner when using in a Fragment

}
```

4. Start capturing live audio from the device's microphone and classify in realtime:

```kotlin
soundClassifier.start()
```

5. Receive classification result in realtime as a map of human-readable class name and 
probabilities of the current sound belonging to each particular category.

```kotlin
let labelName = soundClassifier.labelList[0] // e.g. "Clap"
soundClassifier.probabilities.observe(this) { resultMap ->
    let probability = resultMap[labelName] // e.g. 0.7
}
```
