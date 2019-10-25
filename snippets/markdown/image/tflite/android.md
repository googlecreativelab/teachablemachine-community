The TFLite mobilenet example works with the teachable machine model.

**1.** Get the project from [github](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android)

**2.** Put your downloaded model into `app/src/assets`

**3.** Create/update `labels.txt` in `app/src/assets` to contain each of your classes

**4.** Change the default model type in `CameraActivity.java` in
`android/app/src/main/java/org/tensorflow/lite/examples/classification/`

from

```java
    private Model model = Model.QUANTIZED;
```

to

```java
    private Model model = Model.FLOAT;
```

**5.** Change `ClassifierFloatMobileNet.java` in
`android/app/src/main/java/org/tensorflow/lite/examples/classification/tflite/`

from

```java
    return "mobilenet_v1_1.0_224.tflite";
```

to

```java
    return "model_unquant.tflite";
```
