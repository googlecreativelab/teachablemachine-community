# Teachable Machine Library - Pose

Library for using pose models created with Teachable Machine.

### Model checkpoints

There is one link related to your model that will be provided by Teachable Machine

`https://teachablemachine.withgoogle.com/models/MODEL_ID/`

Which you can use to access:

* The model topology: `https://teachablemachine.withgoogle.com/models/MODEL_ID/model.json`
* The model metadata: `https://teachablemachine.withgoogle.com/models/MODEL_ID/metadata.json`


## Usage

There are two ways to easily use the model provided by Teachable Machine in your Javascript project: by using this library via script tags or by installing this library from NPM (and using a build tool ike Parcel, WebPack, or Rollup)

### via Script Tag

```js
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@teachablemachine/pose@0.8.3/dist/teachablemachine-pose.min.js"></script>
```

### via NPM

[NPM Package](https://www.npmjs.com/package/@teachablemachine/pose)


```
npm i @tensorflow/tfjs
npm i @teachablemachine/pose
```

```js
import * as tf from '@tensorflow/tfjs';
import * as tmPose from '@teachablemachine/pose';
```

### Sample snippet

```js
<div>Teachable Machine Pose Model</div>
<button type='button' onclick='init()'>Start</button>
<div><canvas id='canvas'></canvas></div>
<div id='label-container'></div>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@teachablemachine/pose@0.8.3/dist/teachablemachine-pose.min.js"></script>
<script type="text/javascript">
    // More API functions here:
    // https://github.com/googlecreativelab/teachablemachine-community/tree/master/libraries/pose

    // the link to your model provided by Teachable Machine export panel
    const URL = '{{URL}}';
    let model, webcam, ctx, labelContainer, maxPredictions;

    async function init() {
        const modelURL = URL + 'model.json';
        const metadataURL = URL + 'metadata.json';

        // load the model and metadata
        // Refer to tmPose.loadFromFiles() in the API to support files from a file picker
        model = await tmPose.load(modelURL, metadataURL);
        maxPredictions = model.getTotalClasses();

        // Convenience function to setup a webcam
        const flip = true; // whether to flip the webcam
        webcam = new tmPose.Webcam(200, 200, flip); // width, height, flip
        await webcam.setup(); // request access to the webcam
        webcam.play();
        window.requestAnimationFrame(loop);

        // append/get elements to the DOM
        const canvas = document.getElementById('canvas');
        canvas.width = 200; canvas.height = 200;
        ctx = canvas.getContext('2d');
        labelContainer = document.getElementById('label-container');
        for (let i = 0; i < maxPredictions; i++) { // and class labels
            labelContainer.appendChild(document.createElement('div'));
        }
    }

    async function loop(timestamp) {
        webcam.update(); // update the webcam frame
        await predict();
        window.requestAnimationFrame(loop);
    }

    async function predict() {
        // Prediction #1: run input through posenet
        // estimatePose can take in an image, video or canvas html element
        const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);
        // Prediction 2: run input through teachable machine classification model
        const prediction = await model.predict(posenetOutput);

        for (let i = 0; i < maxPredictions; i++) {
            const classPrediction =
                prediction[i].className + ': ' + prediction[i].probability.toFixed(2);
            labelContainer.childNodes[i].innerHTML = classPrediction;
        }

        // finally draw the poses
        drawPose(pose);
    }

    function drawPose(pose) {
        ctx.drawImage(webcam.canvas, 0, 0);
        // draw the keypoints and skeleton
        if (pose) {
            const minPartConfidence = 0.5;
            tmPose.drawKeypoints(pose.keypoints, minPartConfidence, ctx);
            tmPose.drawSkeleton(pose.keypoints, minPartConfidence, ctx);
        }
    }
</script>
```


## API

### Loading the model - url checkpoints

`tmPose` is the module name, which is automatically included when you use the `<script src>` method. It gets added as an object to your window so you can access via `window.tmPose` or simply `tmPose`.

```ts
tmPose.load(
    checkpoint: string, 
    metadata?: string | Metadata
)
```

Args:

* **checkpoint**: a URL to a json file that contains the model topology and a reference to a bin file (model weights)
* **metadata**: a URL to a json file that contains the text labels of your model and additional information


Usage:

```js
await tmPose.load(checkpointURL, metadataURL);
```

### Loading the model - browser files

You can upload your model files from a local hard drive by using a file picker and the File interface. 

```ts
tmPose.loadFromFiles(
	model: File, 
	weights: File, 
	metadata: File
) 
```

Args:

* **model**: a File object that contains the model topology (.json)
* **weights**: a File object with the model weights (.bin)
* **metadata**: a File object that contains the text labels of your model and additional information (.json)

Usage:

```js
// you need to create File objects, like with file input elements (<input type="file" ...>)
const uploadModel = document.getElementById('upload-model');
const uploadWeights = document.getElementById('upload-weights');
const uploadMetadata = document.getElementById('upload-metadata');
model = await tmPose.loadFromFiles(uploadModel.files[0], uploadWeights.files[0], uploadMetadata.files[0])
```

### Model - get total classes

Once you have loaded a model, you can obtain the total number of classes in the model. 

This method exists on the model that is loaded from `tmPose.load`.

```ts
model.getTotalClasses()
```

Returns a number representing the total number of classes

### Posenet model - estimatePose

You'll have to run your input through two models to make a prediction: first through posenet and then through the classification model created via Teachable Machine.

This method exists on the model that is loaded from `tmPose.load`.

```ts
model.estimatePose(
    sample: ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | tf.Tensor3D,
    flipHorizontal = false
)
```

Args:

* **sample**: an image, canvas, or video to pass through posenet
* **flipHorizontal**: a boolean to trigger whether to flip on X the pose keypoints

Usage:

```js
const flipHorizontal = false;
const { pose, posenetOutput } = await model.estimatePose(webcamElement, flipHorizontal);
```

The function returns `pose` an object with the keypoints data (for drawing) and `posenetOutput` a Float32Array of concatenated posenet output data (for the classification prediction). 

### Teachable Machine model - predict

Once you have the output from posenet, you can make a classificaiton with the Teachable Machine model you trained.

This method exists on the model that is loaded from `tmPose.load`.

```ts
model.predict(
    poseOutput: Float32Array
)
```

Args:

* **poseOutput**: an array representing the output of posenet from the `mode.estimatePose` function

Usage:

```js
// predict can take in an image, video or canvas html element
// if using the webcam utility, we set flip to true since the webcam was only 
// flipped in CSS
const flipHorizontal = false;

const { pose, posenetOutput } = await model.estimatePose(webcamElement, flipHorizontal);
const prediction = await model.predict(posenetOutput);
```



### Teachable Machine model - predictTopK

An alternative function to `predict()` which returns probabilities for all classes.

This method exists on the model that is loaded from `tmPose.load`.

```ts
model.predictTopK(
    poseOutput: Float32Array,
    maxPredictions = 10
)
```

Args:

* **poseOutput**: an array representing the output of posenet from the `mode.estimatePose` function
* **maxPredictions**: total number of predictions to return

Usage:

```js
// predictTopK can take in an image, video or canvas html element
// if using the webcam utility, we set flip to true since the webcam was only 
// flipped in CSS
const maxPredictions = model.getTotalClasses();
const flipHorizontal = false;

const { pose, posenetOutput } = await model.estimatePose(webcamElement, flipHorizontal);
const prediction = await model.predictTopK(posenetOutput, maxPredictions);
```

### Webcam

You can optionally use a webcam class that comes with the library, or spin up your own webcam. This class exists on the `tmPose` module.

Please note that the default webcam used in Teachable Machine was flipped on X - so you should probably set `flip = true` if creating your own webcam unless you flipped it manually in Teachable Machine.

```ts
new tmPose.Webcam(
    width = 400,
    height = 400,
    flip = false,
)
```

Args:

* **width**: width of the webcam. It should ideally be square since that's how the model was trained with Teachable Machine.
* **height**: height of the webcam. It should ideally be square since that's how the model was trained with Teachable Machine.
* **flip**: boolean to signal whether webcam should be flipped on X. Please note this is only flipping on CSS.

Usage:

```js
// webcam has a square ratio and is flipped by default to match training
const webcam = new tmPose.Webcam(200, 200, true);
await webcam.setup();
webcam.play();
document.body.appendChild(webcam.canvas);
```

### Webcam - setup

After creating a Webcam object you need to call setup just once to set it up.

```ts
webcam.setup(
	options: MediaTrackConstraints = {}
)
```

Args:

* **options**: optional media track contraints for the webcam

Usage:

```js
await webcam.setup();
```

### Webcam - play, pause, stop

```ts
webcam.play();
webcam.pause();
webcam.stop();
```

Webcam play loads and starts playback of a media resource. Returns a promise.

### Webcam - update

Call on update to update the webcam frame.

```ts
webcam.update();
```


### Draw keypoints

You can optionally use a utility function to draw the pose keypoints from `model.estimatePose`.

```ts
tmPose.drawKeypoints(
    keypoints: Keypoint[], 
    minConfidence: number, 
    ctx: CanvasRenderingContext2D, 
    keypointSize: number = 4, 
    fillColor: string = 'aqua', 
    strokeColor: string = 'aqua', 
    scale = 1
)
```

Args:

* **keypoints**: keypoints array
* **minConfidence**: will not draw keypoints below this confidence score
* **ctx**: canvas to draw on
* **keypointsSize**: size of the keypoints for drawing
* **fillColor**: css fill color
* **strokeColor**: css stroke colo
* **scale**: a scale factor for the drawing

Usage:

```js
const flipHorizontal = false;
const { pose, posenetOutput } = await model.estimatePose(webcamEl, flipHorizontal);

const minPartConfidence = 0.5;
tmPose.drawKeypoints(pose.keypoints, minPartConfidence, canvasContext);
```

### Draw skeleton

You can optionally use a utility function to draw the pose keypoints from `model.estimatePose`.

```ts
tmPose.drawSkeleton(
    keypoints: Keypoint[], 
    minConfidence: number, 
    ctx: CanvasRenderingContext2D, 
    lineWidth: number = 2, 
    strokeColor: string = 'aqua', 
    scale = 1
)
```

Args:

* **keypoints**: keypoints array
* **minConfidence**: will not draw keypoints below this confidence score
* **ctx**: canvas to draw on
* **lineWidth**: width of the segment lines to draw
* **strokeColor**: css stroke colo
* **scale**: a scale factor for the drawing

Usage:

```js
const flipHorizontal = false;
const { pose, posenetOutput } = await model.estimatePose(webcamEl, flipHorizontal);

const minPartConfidence = 0.5;
tmPose.drawKeypoints(pose.keypoints, minPartConfidence, canvasContext);
tmPose.drawSkeleton(pose.keypoints, minPartConfidence, canvasContext);
```
