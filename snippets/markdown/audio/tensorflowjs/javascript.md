```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.1.2/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/speech-commands@0.3.8/dist/speech-commands.min.js"></script>

<script type="text/javascript">
    // more documentation available at
    // https://github.com/tensorflow/tfjs-models/tree/master/speech-commands
    
    // the json file (model topology) has a reference to the bin file (model weights)
    const checkpointURL = '{{MODEL_URL}}';
    // the metatadata json file contains the text labels of your model and additional information
    const metadataURL = '{{METADATA_URL}}';

    const recognizer = speechCommands.create(
        'BROWSER_FFT',
        undefined,
        checkpointURL,
        metadataURL);


    async function setup() {
        // Make sure that the underlying model and metadata are loaded via HTTPS
        // requests.
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
        // - result.scores contains the probability scores that correspond to
        //   recognizer.wordLabels().
        // - result.spectrogram contains the spectrogram of the recognized word.
            console.log(result);
        }, {
            includeSpectrogram: true,
            probabilityThreshold: 0.75,
            invokeCallbackOnNoiseAndUnknown: true,
            overlapFactor: 0.50 // probably want between 0.5 and 0.75. More info in README
        });

        // Stop the recognition in 10 seconds.
        // setTimeout(() => recognizer.stopListening(), 10e3);
    }

    setup();
</script>
```