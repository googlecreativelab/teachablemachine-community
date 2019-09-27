Use this code snippet to use this model:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.1.2/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@0.6/dist/teachablemachine-image.min.js"></script>
<script type="text/javascript">
    // The json file defining the weights of the model
    const checkpointURL = '{{MODEL_URL}}';

    // The metatadata json file contains the text labels of your model
    // and additional information
    const metadataURL = '{{METADATA_URL}}';

    let model, webcamEl, maxPredictions;

    async function init() {
        // load the model and metadata
        // Refer to tmImage.loadFromFiles() in the API to support files from a file picker
        // or files from your local hard drive
        model = await tmImage.load(checkpointURL, metadataURL);
        maxPredictions = model.getTotalClasses();

        // optional function for creating a webcam
        // webcam has a square ratio and is flipped by default to match training
        const webcamFlipped = true;
        webcamEl = await tmImage.getWebcam(200, 200, 'front', webcamFlipped);
        webcamEl.play();
        document.body.appendChild(webcamEl);

        window.requestAnimationFrame(loop); // kick of pose prediction loop
    }

    async function loop(timestamp) {
        await predict();
        window.requestAnimationFrame(loop);
    }

    async function predict() {
        // predict can take in an image, video or canvas html element
        // we set flip to true since the webcam was only flipped in CSS
        const flip = true;
        const prediction = await model.predict(webcamEl, flip, maxPredictions);
        console.log(prediction);
    }

    init();
</script>
```

Learn more about how to use the code snippet on [github](https://github.com/googlecreativelab/teachablemachine-libraries)
