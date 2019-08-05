# Teachable Machine Library - Image

Library for using image models created with Teachable Machine.

### Model checkpoints

There are two links related to your model that will be provided by Teachable Machine:

1) The model topology: `https://storage.googleapis.com/tm-mobilenet/YOUR_MODEL_NAME/model.json`

2) and a metadata JSON file: `https://storage.googleapis.com/tm-mobilenet/YOUR_MODEL_NAME/metadata.json`


## Usage

There are two ways to easily use the model provided by Teachable Machine in your Javascript project: by using this library via script tags or by installing this library from NPM (and using a build tool ike Parcel, WebPack, or Rollup)

### via Script Tag

```js
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.1.2/dist/tf.min.js"></script>
<script src="https://storage.googleapis.com/tm-pro/v0.2.0/teachablemachine-image.min.js"></script>
```

### via NPM

Coming soon

### Sample snippet

```js
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.1.2/dist/tf.min.js"></script>
<script src="https://storage.googleapis.com/tm-pro/v0.2.0/teachablemachine-image.min.js"></script>
<script type="text/javascript">
    const checkpointURL = 'https://storage.googleapis.com/tm-mobilenet/YOUR_MODEL_NAME/model.json';
    const metadataURL = 'https://storage.googleapis.com/tm-mobilenet/YOUR_MODEL_NAME/metadata.json';

    let model;
    let webcamEl;

    async function init() {
        model = await tmImage.mobilenet.load(checkpointURL, metadataURL);
        const maxPredictions = model.getTotalClasses();

        // webcam has a square ratio and is flipped by default to match training
        webcamEl = await tmImage.getWebcam(200, 200, ‘front’);
        webcamEl.play();
        document.body.appendChild(webcamEl);

        // predict can take in an image, video or canvas html element
        // we set flip to true since the webcam was only flipped in CSS
        const flip = true;
        const prediction = await model.predict(webcamEl, flip, maxPredictions);
        console.log(prediction);
    }

    init();
</script>
```


## API

### Loading the model - url checkpoints

`tmImage.mobilenet` is the module name, which is automatically included when you use the `<script src>` method. When using ES6 imports, `mobilenet` is the module.

```ts
tmImage.mobilenet.load(
	checkpoint: string, 
	metadata?: string | Metadata
)
```

Args:

* **checkpoint**: a URL to a json file that contains the model topology and a reference to a bin file (model weights)
* **metadata**: a URL to a json file that contains the text labels of your model and additional information


Usage:

```js
await tmImage.mobilenet.load(checkpointURL, metadataURL);
```


### Loading the model - browser files

You can upload your model files from a local hard drive by using a file picker and the File interface. 

```ts
tmImage.mobilenet.loadFromFiles(
	model: File, 
	weights: File, 
	metadata?: string | Metadata
) 
```

Args:

* **model**: a File object that contains the model topology (.json)
* **weights**: a File object with the model weights (.bin)
* **metadata**: a File object that contains the text labels of your model and additional information (.json)

Usage:

```js
// you need to create File objects, like with file input elements (<input type="file" ...>)
const uploadJSONInput = document.getElementById('upload-json');
const uploadWeightsInput = document.getElementById('upload-weights');
model = await tmImage.mobilenet.loadFromFiles(uploadJSONInput.files[0], uploadWeightsInput.files[0])
```

### Model - get total classes

Once you have loaded a model, you can obtain the total number of classes in the model. 

This method exists on the model that is loaded from `tmImage.mobilenet.load`.

```ts
model.getTotalClasses()
```

Returns a number representing the total number of classes


### Model - predict

Once you have loaded a model, you can make a classificaiton with a couple of different input options.

This method exists on the model that is loaded from `tmImage.mobilenet.load`.

```ts
model.predict(
  image: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | ImageBitmap,
  flipped = false,
  maxPredictions = 10
)
```

Args:

* **image**: an image, canvas, or video element to make a classification on
* **flipped**: a boolean to trigger whether to flip on X or not the image input
* **maxPredictions**: total number of predictions to return

Usage:

```js
// predict can take in an image, video or canvas html element
// if using the webcam utility, we set flip to true since the webcam was only 
// flipped in CSS
const flip = true;
const maxPredictions = model.getTotalClasses();
const prediction = await model.predict(webcamElement, flip, maxPredictions);
```

### Webcam

You can optionally use a webcam utility that comes with the library, or spin up your own webcam. This method exists on the `tmImage` module.

Please note that the webcam used in Teachable Machine was flipped on X - so you should probably do the same if creating your own webcam.

```ts
tmImage.getWebcam(
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
const webcamEl = await tmImage.getWebcam(200, 200);
webcamEl.play();
document.body.appendChild(webcamEl);
```

or

```js
const webcamEl = await tmImage.getWebcam(200, 200, 'front');
const webcamEl = await tmImage.getWebcam(200, 200, 'back');
const webcamEl = await tmImage.getWebcam(200, 200, 'front', false);
```



