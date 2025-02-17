Learn more about how to use the code snippet on [github](https://github.com/googlecreativelab/teachablemachine-community/tree/master/libraries/image).

```html
<div>Teachable Machine Image Model</div>
<button type="button" onclick="init()">Start</button>
<button type="button" onclick="stopWebcam()">Stop</button>
<div id="status">Initializing...</div>
<div id="webcam-container"></div>
<div id="label-container"></div>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@latest/dist/teachablemachine-image.min.js"></script>
<script type="text/javascript">
    // More API functions here:
    // https://github.com/googlecreativelab/teachablemachine-community/tree/master/libraries/image

    // the link to your model provided by Teachable Machine export panel
    const URL = "{{URL}}";

    let model, webcam, labelContainer, maxPredictions, isRunning = false;

    // Load the image model and setup the webcam
    async function init() {

        document.getElementById("status").innerText = "Loading model...";

        const modelURL = URL + "model.json";
        const metadataURL = URL + "metadata.json";

        // load the model and metadata
        // Refer to tmImage.loadFromFiles() in the API to support files from a file picker
        // or files from your local hard drive
        // Note: the pose library adds "tmImage" object to your window (window.tmImage)
        model = await tmImage.load(modelURL, metadataURL);
        maxPredictions = model.getTotalClasses();

        // Convenience function to setup a webcam
        const flip = true; // whether to flip the webcam
        webcam = new tmImage.Webcam(200, 200, flip); // width, height, flip
        await webcam.setup(); // request access to the webcam
        await webcam.play();
        isRunning = true;
        window.requestAnimationFrame(loop);

        // append elements to the DOM
        document.getElementById("webcam-container").innerHTML = "";
        document.getElementById("webcam-container").appendChild(webcam.canvas);

        labelContainer = document.getElementById("label-container");
        labelContainer.innerHTML = "";
        for (let i = 0; i < maxPredictions; i++) {
            let label = document.createElement("div");
            label.style.fontSize = "16px";
            label.style.marginTop = "5px";
            label.style.fontWeight = "bold";
            labelContainer.appendChild(label);
        }

        document.getElementById("status").innerText = "Model loaded. Ready!";
    }

    async function loop() {
        webcam.update(); // update the webcam frame
        await predict();
        window.requestAnimationFrame(loop);
    }

    // run the webcam image through the image model
    async function predict() {
        // predict can take in an image, video or canvas html element
        const prediction = await model.predict(webcam.canvas);
        for (let i = 0; i < maxPredictions; i++) {
            const classPrediction = `${prediction[i].className}: ${Math.round(prediction[i].probability * 100)}%`;
            labelContainer.childNodes[i].innerHTML = classPrediction;
        }
    }

    // stop webcam execution
    function stopWebcam() {
            if (webcam) {
                webcam.stop();
                isRunning = false;
                document.getElementById("status").innerText = "Webcam stopped.";
            }
        }
</script>
```
