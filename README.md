# Teachable Machine Support Library

This repo contains the support library for a new version of Teachable Machine. For more info go to: [Teachable Machine](https://teachablemachine.withgoogle.com/io19).

# Images

## Model checkpoints

There are two links related to your model that will be provided by Teachable Machine:

The model topology:

`https://storage.googleapis.com/tm-mobilenet/YOUR_MODEL_NAME/model.json`

and a metadata JSON file:

`https://storage.googleapis.com/tm-mobilenet/YOUR_MODEL_NAME/metadata.json`

## Using the image support library

Include the following script tags to use an image model from Teachable Machine:

```
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.1.2/dist/tf.min.js"></script>
<script src="https://storage.googleapis.com/tm-pro/v1.0.5-c/teachablemachine-image.min.js"></script>
```

## Sample snippet

```
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.1.2/dist/tf.min.js"></script>
<script src="https://storage.googleapis.com/tm-pro/v1.0.5-c/teachablemachine-image.min.js"></script>
<script type="text/javascript">
    // the json file (model topology) has a reference to the bin file (model weights)
    const checkpointURL = 'https://storage.googleapis.com/tm-mobilenet/YOUR_MODEL_NAME/model.json';
    // the metatadata json file contains the text labels of your model and additional information
    const metadataURL = 'https://storage.googleapis.com/tm-mobilenet/YOUR_MODEL_NAME/metadata.json';

    let model;
    let webcamEl;

    async function init() {
        // load the model and metadata
        model = await tm.mobilenet.load(checkpointURL, metadataURL);
        const maxPredictions = model.getTotalClasses();

        // optional function for creating a webcam
        // webcam has a square ratio and is flipped by default to match training
        webcamEl = await tm.getWebcam(200, 200, ‘front’);
        webcamEl.play();
        document.body.appendChild(webcamEl);

        // use tm.mobilenet.loadFromFiles() function to support files from a file picker or files from your local hard drive
        // you need to create File objects, like with file input elements (<input type="file" ...>)
        // const uploadJSONInput = document.getElementById('upload-json');
        // const uploadWeightsInput = document.getElementById('upload-weights');
        // model = await tm.mobilenet.loadFromFiles(uploadJSONInput.files[0], uploadWeightsInput.files[0])

        // predict can take in an image, video or canvas html element
        // we set flip to true since the webcam was only flipped in CSS
        const flip = true;
        const prediction = await model.predict(webcamEl, flip, maxPredictions);
        console.log(prediction);
    }

    init();
</script>
```



# Audio

## Model checkpoints

There are two links related to your model that will be provided by Teachable Machine:

The model topology:

`https://storage.googleapis.com/tm-speech-commands/YOUR_MODEL_NAME/model.json`

and a metadata JSON file:

`https://storage.googleapis.com/tm-speech-commands/YOUR_MODEL_NAME/metadata.json`

## Using the audio support library

Include the following script tags to use an audio model from Teachable Machine:

```
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.1.2/dist/tf.min.js">
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/speech-commands@0.3.8/dist/speech-commands.min.js">
```


## API Reference

More information avaiable here: [Speech Commands Model](https://github.com/tensorflow/tfjs-models/tree/master/speech-commands)


## Sample snippet

```
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.1.2/dist/tf.min.js">
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/speech-commands@0.3.8/dist/speech-commands.min.js">

<script type="text/javascript">
    // The json file (model topology) has a reference to the bin file (model weights) 
    // If loading a local json, it will reference a relative path to the bin file
    const modelJson = 'https://storage.googleapis.com/tm-speech-commands/YOUR_MODEL_NAME/model.json';
    const metadataJson = 'https://storage.googleapis.com/tm-speech-commands/YOUR_MODEL_NAME/metadata.json';

    const recognizer = speechCommands.create(
        'BROWSER_FFT',
        undefined,
        modelJson,
        metaDataJson);

    // Make sure that the underlying model and metadata are loaded via HTTPS requests.
    await recognizer.ensureModelLoaded();

    // See the array of words that the recognizer is trained to recognize.
    console.log(recognizer.wordLabels());

    // listen() takes two arguments:
    // 1. A callback function that is invoked anytime a word is recognized.
    // 2. A configuration object with adjustable fields such a
    //    - includeSpectrogram
    //    - probabilityThreshold
    //    - includeEmbedding
    recognizer.listen(result => {
    // - result.scores contains the probability scores that correspond to recognizer.wordLabels().
    // - result.spectrogram contains the spectrogram of the recognized word.
    }, {
    includeSpectrogram: true,
    probabilityThreshold: 0.75,
    overlapFactor: 0.5
    });

    // Stop the recognition in 10 seconds.
    setTimeout(() => recognizer.stopListening(), 10e3);
</script>
```


# Disclaimer

This is an experiment, not an official Google product. We’ll do our best to support and maintain this experiment but your mileage may vary.

We encourage open sourcing projects as a way of learning from each other. Please respect our and other creators’ rights, including copyright and trademark rights when present, when sharing these works and creating derivative work. If you want more info on Google's policy, you can find that [here](https://www.google.com/permissions/).