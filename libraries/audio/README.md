# Teachable Machine Library - Audio

Library for using audio models created with Teachable Machine.

### Model checkpoints

There are two links related to your model that will be provided by Teachable Machine:

1) The model topology: `https://storage.googleapis.com/tm-speech-commands/YOUR_MODEL_NAME/model.json`

2) and a metadata JSON file: `https://storage.googleapis.com/tm-speech-commands/YOUR_MODEL_NAME/metadata.json`



## Usage

There are two ways to easily use the model provided by Teachable Machine in your Javascript project: by using this library via script tags or by installing this library from NPM (and using a build tool ike Parcel, WebPack, or Rollup)


### via Script Tag

```js
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.1.2/dist/tf.min.js">
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/speech-commands@0.3.8/dist/speech-commands.min.js">
```

### via NPM

Coming soon

### Sample snippet

```js
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.1.2/dist/tf.min.js">
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/speech-commands@0.3.8/dist/speech-commands.min.js">

<script type="text/javascript">
    const modelJson = 'https://storage.googleapis.com/tm-speech-commands/YOUR_MODEL_NAME/model.json';
    const metadataJson = 'https://storage.googleapis.com/tm-speech-commands/YOUR_MODEL_NAME/metadata.json';

    const recognizer = speechCommands.create(
        'BROWSER_FFT',
        undefined,
        modelJson,
        metaDataJson);

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

## API

Please refer to the [Speech Commands](https://github.com/tensorflow/tfjs-models/tree/master/speech-commands) model documentation for more details about the API. 
