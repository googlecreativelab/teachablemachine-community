```html
<div>Teachable Machine Image Model - p5.js and ml5.js</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/0.9.0/p5.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/0.9.0/addons/p5.dom.min.js"></script>
<script src="https://unpkg.com/ml5@0.4.1/dist/ml5.min.js"></script>
<script type="text/javascript">
  // Classifier Variable
  let classifier;
  // Model URL
  const imageModel = '{{URL}}';

  // Video
  let video;

  // To store the classification
  let label = "";
  
  const size = 240;

  // Load the model first
  function preload() {
    classifier = ml5.imageClassifier(imageModel + 'model.json');
  }

  function setup() {
    const constraints = {
      audio: false,
      video: {
        aspectRatio: 1
      }
    };
  
    createCanvas(size, size);
    // Create the video
    video = createCapture(constraints);
    video.size(size, size);
    video.hide();

    // Start classifying
    classifyVideo();
  }

  function draw() {
    background(0);
    // Draw the video
    image(video, 0, 0);

    // Draw the label
    fill(255);
    textSize(16);
    textAlign(CENTER);
    text(label, width / 2, height - 4);
  }

  // Get a prediction for the current video frame
  function classifyVideo() {
    classifier.classify(video, gotResult);
  }

  // When we get a result
  function gotResult(error, results) {
    // If there is an error
    if (error) {
      console.error(error);
      return;
    }
    // The results are in an array ordered by confidence.
    console.log(results[0]);
    label = results[0].label;
    // Classifiy again!
    classifyVideo();
  }
</script>
```
