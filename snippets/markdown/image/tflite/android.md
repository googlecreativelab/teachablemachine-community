For this Teachable Machine example, the _Quantized_ tflite model is being used.
It is using the [TFLite Android example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android),
note that the example only supports models with 3 or more classes,
even though the classifier itself in the example supports 2.

**1.** Get the Android app example from [Github](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android)

**2.** Unpack the _converted_tflite_quantized.zip_ archive exported from Teachable Machine

**3.** Copy _converted_tflite_quantized_ folder to the example asset folder `examples/lite/examples/image_classification/android/app/src/main/assets/`

**4.** Open [`examples/lite/examples/image_classification/android/app/src/main/java/org/tensorflow/lite/examples/classification/tflite/_ClassifierQuantizedMobileNet.java`](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/app/src/main/java/org/tensorflow/lite/examples/classification/tflite/ClassifierQuantizedMobileNet.java)

**5.** Modify `getModelPath()` and `getLabelPath()` to

```java
@Override
protected String getModelPath() {
  return "converted_tflite_quantized/model.tflite";
}

@Override
protected String getLabelPath() {
  return "converted_tflite_quantized/labels.txt";
}
```

You can now build the app using _Android Studio_ as described in the [README](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md).

To enable your model in the app, switch the active model to _Quantized_MobileNet_
