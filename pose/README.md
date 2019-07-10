# Teachable Machine Library - Pose

Library for using pose models created with Teachable Machine.

### Model checkpoints

There are two links related to your model that will be provided by Teachable Machine:

1) The model topology: `https://storage.googleapis.com/tm-posenet/YOUR_MODEL_NAME/model.json`

2) and a metadata JSON file: `https://storage.googleapis.com/tm-posenet/YOUR_MODEL_NAME/metadata.json`


## Usage

There are two ways to easily use the model provided by Teachable Machine in your Javascript project: by using this library via script tags or by installing this library from NPM (and using a build tool ike Parcel, WebPack, or Rollup)

### via Script Tag

```js
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.1.2/dist/tf.min.js"></script>
<script src="https://storage.googleapis.com/tm-pro/v0.1.0/teachablemachine-pose.min.js"></script>
```

### via NPM

Coming soon

### Sample snippet

```js
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.1.2/dist/tf.min.js"></script>
<script src="https://storage.googleapis.com/tm-pro/{{version}}/teachablemachine-pose.min.js"></script>
<script type="text/javascript">
    // the json file (model topology) has a reference to the bin file (model weights)
    const checkpointURL = 'https://storage.googleapis.com/tm-posenet/YOUR_MODEL_NAME/model.json';
    // the metatadata json file contains the text labels of your model and additional information
    const metadataURL = 'https://storage.googleapis.com/tm-posenet/YOUR_MODEL_NAME/metadata.json';

    let model; let webcamEl; let ctx; let maxPredictions;
       
    async function init() {
        // load the model and metadata
        model = await tm.posenet.load(checkpointURL, metadataURL);
        maxPredictions = model.getTotalClasses();

        const width = 200; const height = 200;

        // optional function for creating a webcam
        // webcam has a square ratio and is flipped by default to match training
        webcamEl = await tm.getWebcam(width, height);
        webcamEl.play();
        // document.body.appendChild(webcamEl);
        
        // optional function for creating a canvas to draw the webcam + keypoints to
        const flip = true;
        const canvas = tm.createCanvas(width, height, flip);
        ctx = canvas.getContext('2d');
        document.body.appendChild(canvas);
        
        window.requestAnimationFrame(loop); // kick of pose prediction loop
    }
    
    async function loop(timestamp) {
        await predict();
        window.requestAnimationFrame(loop);
    }
    
    async function predict() {
        // Prediction #1: run input through posenet
        // predictPosenet can take in an image, video or canvas html element
        const flipHorizontal = false;
        const { pose, posenetOutput } = await model.predictPosenet(webcamEl, flipHorizontal);
        // Prediction 2: run input through teachable machine assification model
        const prediction = await model.predict(posenetOutput, flipHorizontal, maxPredictions);

        ctx.drawImage(webcamEl, 0, 0);
        // draw the keypoints and skeleton
        if (pose) {
            const minPartConfidence = 0.5;
            tm.drawKeypoints(pose.keypoints, minPartConfidence, ctx);
            tm.drawSkeleton(pose.keypoints, minPartConfidence, ctx);
        }

        console.log(prediction);      
    }

    init();
</script>

```


## API

### Loading the model - url checkpoints

`tm.posenet` is the module name, which is automatically included when you use the `<script src>` method. When using ES6 imports, `posenet` is the module.

```ts
tm.posenet.load(
    checkpoint: string, 
    metadata?: string | Metadata
)
```

Args:

* **checkpoint**: a URL to a json file that contains the model topology and a reference to a bin file (model weights)
* **metadata**: a URL to a json file that contains the text labels of your model and additional information


Usage:

```js
await tm.posenet.load(checkpointURL, metadataURL);
```

### Model - get total classes

Once you have loaded a model, you can obtain the total number of classes in the model. 

This method exists on the model that is loaded from `tm.posenet.load`.

```ts
model.getTotalClasses()
```

Returns a number representing the total number of classes

### Posenet model - predict

You'll have to run your input through two models to make a prediction: first through posenet and then through the classification model created via Teachable Machine.

This method exists on the model that is loaded from `tm.posenet.load`.

```ts
model.predictPosenet(
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
const { pose, posenetOutput } = await model.predictPosenet(webcamElement, flipHorizontal);
```

The function returns `pose` an object with the keypoints data (for drawing) and `posenetOutput` a Float32Array of concatenated posenet output data (for the classification prediction). 


### Teachable Machine model - predict

Once you have the output from posenet, you can make a classificaiton with the Teachable Machine model you trained.

This method exists on the model that is loaded from `tm.posenet.load`.

```ts
model.predict(
    poseOutput: Float32Array,
    flipped = false,
    maxPredictions = 10
)
```

Args:

* **poseOutput**: an array representing the output of posenet from the `mode.predictPosenet` function
* **flipped**: a boolean to trigger whether to flip on X or not the image input
* **maxPredictions**: total number of predictions to return

Usage:

```js
// predict can take in an image, video or canvas html element
// if using the webcam utility, we set flip to true since the webcam was only 
// flipped in CSS
const maxPredictions = model.getTotalClasses();
const flipHorizontal = false;

const { pose, posenetOutput } = await model.predictPosenet(webcamElement, flipHorizontal);
const prediction = await model.predict(posenetOutput, flipHorizontal, maxPredictions);
```

### Webcam

You can optionally use a webcam utility that comes with the library, or spin up your own webcam. This method exists on the `tm` module.

Please note that the webcam used in Teachable Machine was flipped on X - so you should probably do the same if creating your own webcam.

```ts
tm.getWebcam(
    width = 400,
    height = 400,
    facingMode = 'front',
    flipped = true,
    video: HTMLVideoElement = document.createElement('video'),
    options: MediaTrackConstraints = defaultConstraints
)
```

Args:

* **width**: width of the webcam. It should ideally be square since that's how the model was trained with Teachable Machine.
* **height**: height of the webcam. It should ideally be square since that's how the model was trained with Teachable Machine.
* **facingMode**: 'front' or 'back'. Whether to use the front or back facing camera.
* **flipped**: boolean to signal whether webcam should be flipped on X. Please note this is only flipping on CSS.
* **video**: video element for the webcam
* **options**: video constraints

Usage:

```js
// webcam has a square ratio and is flipped by default to match training
const webcamEl = await tm.getWebcam(200, 200);
webcamEl.play();
document.body.appendChild(webcamEl);
```

or

```js
const webcamEl = await tm.getWebcam(200, 200, 'front');
const webcamEl = await tm.getWebcam(200, 200, 'back');
const webcamEl = await tm.getWebcam(200, 200, 'front', false);
```

### Create canvas

You can optionally use a utility to create a canvas for drawing the webcam data as well as the pose keypoints.

```ts
tm.createCanvas(
    width = 200,
    height = 200,
    flipHorizontal = false
)
```

Args:

* **width**: width of the canvas.
* **height**: height of the canvas
* **flipHorizontal**: boolean to signal whether canvas should be flipped on X for drawing.

Usage:

```js
const flip = false;
const canvas = tm.createCanvas(200, 200, flip);
```

### Draw keypoints

You can optionally use a utility function to draw the pose keypoints from `model.predictPosenet`.

```ts
tm.drawKeypoints(
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
const { pose, posenetOutput } = await model.predictPosenet(webcamEl, flipHorizontal);

const minPartConfidence = 0.5;
tm.drawKeypoints(pose.keypoints, minPartConfidence, canvasContext);
```

### Draw skeleton

You can optionally use a utility function to draw the pose keypoints from `model.predictPosenet`.

```ts
tm.drawSkeleton(
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
const { pose, posenetOutput } = await model.predictPosenet(webcamEl, flipHorizontal);

const minPartConfidence = 0.5;
tm.drawKeypoints(pose.keypoints, minPartConfidence, canvasContext);
tm.drawSkeleton(pose.keypoints, minPartConfidence, canvasContext);
```