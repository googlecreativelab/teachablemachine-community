For this Teachable Machine example, the _Quantized_ tflite model is being used.
(The teachable machine model works with the TFLite _mobilenet_ example.)

**1.** Get the Android app example from [Github](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android)

**2.** Unpack the _converted_tflite_quantized.zip_ archive exported from Teachable Machine

**3.** Rename _labels.txt_ to _mylabels.txt_ (so there will be no conflict with the existing _labels.txt_ file later on) 

**4.** Rename _model.tflite_ to _modelquantized.tflite_ (so we remember this model is quantized)

**5.** Copy _mylabels.txt_ and _modelquantized.tflite_

**6.** Go to `examples-master\lite\examples\image_classification\android\app\src\main\assets`

**7.** Paste in _mylabels.txt_ and _modelquantized.tflite_

**8.** Go to `examples-master\lite\examples\image_classification\android\app\src\main\java\org\tensorflow\lite\examples\classification\tflite`

**9.** Open _ClassifierQuantizedMobileNet.java_ to edit

**10.** In line 55 replace _mobilenet_v1_1.0_224_quant.tflite_ with your model _modelquantized.tflite_

```
  protected String getModelPath() {
    return "mobilenet_v1_1.0_224_quant.tflite";
  }
```

**11.** In line 60  replace _labels.txt_ with your label file _mylabels.txt_

```
  @Override
  protected String getLabelPath() {
    return "labels.txt";
  }
```

**12.** Save your changes to _ClassifierQuantizedMobileNet.java_

Now you can build the app using _Android Studio_ as described in the [README](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md).

To enable your model in the app, switch the active model to _Quantized_MobileNet_
