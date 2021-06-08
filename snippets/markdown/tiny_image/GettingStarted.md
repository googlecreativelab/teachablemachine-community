# Getting Started With Teachable Machine's Embedded Model

This guide is an overview of how to use Teachable Machine's Embedded Model.

Teachable Machine is a web based tool that lets you train models entirely in the browser with no coding required. If you’ve never used Teachable Machine before, check out the getting started videos here to understand how it works. The embedded model is a smaller version of the standard image model, allowing you to run it on microcontrollers.
This guide will show you how to connect a camera and Arduino Nano 33 BLE to Teachable Machine to gather training samples, how to train your model, and finally how to export the model and run it on device.

## Requirements

This guide is meant for use with the Arduino Nano 33 BLE Sense and Nano 33 BLE, combined with the OV7670 Camera module. You'll need some female to female leads to connect the two.
We found the OV7670 Camera to be the most reliable, but you can also use the Arducam 2MP Plus. by replacing the image_provider.h with the one provided here.
You'll also need a computer to run Teachable Machine, run the Arduino IDE, and run Processing.

## Software Setup

You'll need a few programs and libraries to communicate with your OV7670 Camera.

[Download and install the Arduino IDE from the Arduino Website](https://www.arduino.cc/en/software). Then:

### Install the Arduino_TensorFlowLite library

To install it, open the Arduino library manager in Tools -> Manage Libraries... and search for Arduino_TensorFlowLite. **Make sure to select Version 2.4.0-ALPHA or later and hit install.**

### Install the Arduino_OV767X library

To install it, open the Arduino library manager in Tools -> Manage Libraries... and search for Arduino_OV767X and install this as well.

### Processing IDE

We will use a simple Processing Sketch to connect our arduino to Teachable Machine. [To get this running, first install the Processing IDE from the website here.](https://processing.org/download/)

You'll need a few additional libraries for our processing sketch as well. Open the Processing IDE. In the Menu Bar, go to Sketch -> Add Library -> Manage Libraries.

You'll need to install 2 libraries, the first is the ControlP5 library, and the second is called Websockets. Search for and install them.

![image of processing library manager](GettingStartedImages/processing_libraries.png)

## Hardware Setup

Use female to female leads to connect the Ov7670 Camera and Arduino. Depending on which 0v7670 variant you have, the pin labels may vary slighly, but the layout of the pins will be the same.

|0v7670 Camera Pin Name|Arduino pin name|
|----------------|----------------|
|3.3v|3.3v|
|GND|GND (either pin marked GND is fine)|
|SCL/SIOC|A5|
|SDA/SIOD|A4|
|VS/VSYNC|D8|
|HS/HREF|A1|
|PCLK|A0|
|MCLK/XCLK|D9|
|D7|D4|
|D6|D6|
|D5|D5|
|D4|D3|
|D3|D2|
|D2|D0 / RX|
|D1|D1 / TX|
|D0|D10|

Leave any remaining pins disconnected on the OV7670

## Training

Now that the camera is set up, let's connect it to Teachable Machine so we can use it as an input for your machine learning model.

### Run Uploader and Connector Sketches

Create a new teachable machine [image project here](https://teachablemachine.withgoogle.com) **Make sure to select 'Embedded model' so that you'll be able to export our model to your arduino.** 

Select **Device** as the input type, a window like this should pop up.

![Image of Teachable Machine](./GettingStartedImages/sketches.png)

**First, Download [TMArduinoSketch.zip](https://storage.googleapis.com/tiny-templates/TMConnector.zip)**, unzip it, and open it in the Arduino IDE by double clicking the .ino file. Make sure that you have selected Arduino Nano 33 inside of Tools -> Boards, and the correct port is selected under Tools -> Port. Upload this sketch to the Arduino. This sets up the Arduino to send images to teachable machine.

**Next Download the [TMProcessingSketch.zip](https://storage.googleapis.com/tiny-templates/TMUploader.zip)**, unzip it and open it in the Processing IDE by double clicking on the .pde file. Hit play in the upper left corner. You should see a window Like this come up:

![Image of Processing Sketch](./GettingStartedImages/port_select.png)

Use the port select menu to choose the port for your Arduino. A good way to figure out what this should be is by referencing the port used in the Arduino IDE to upload the arduino sketch

You should see the camera feed appear in the Processing App.

```
Troubleshooting:

- If you see static or nothing coming from the camera, make sure all your wiring connections are tight and correct.
- If you see a flat gray screen, try adjusting the camera's focus so that it's able to correctly auto-expose.
```
Now hit 'Attempt to connect to device' on the Teachable Machine website. You should see a preview of your OV7670 Camera feed in the class window.

## Collect the Data

Use the record button to collect samples inside of each class. The samples should contain examples of what you want to classify in real world lighting.

![Example of training data](GettingStartedImages/train_data.png)

```text
Note: Remember that the results of any machine learning model depend on the examples you give it. Trying different examples is a core part of exploring machine learning. So, if it's not working as you intended, play around with different approaches for what examples you provide.
```

When you're done adding training data, hit Train and don't switch tabs.

## Test Your Model

Try your model out in the teachable machine preview window. Select TinyBLE from the input dropdown menu, you'll see the OV7670 Camera images start to be classified.

![Teachable Machine Preview window](GettingStartedImages/tm_fpo.gif)

## Running your model on the Arduino

Hit Export model above the preview window in Teachable Machine. Select Tensorflow Lite, then Tensorflow Lite for Microcontrollers and Hit 'Download my Model'. This will convert your model in the background for a few moments, then download a zip folder containing an arduino sketch with your model loaded.

*Close any open Processing Sketches* and upload this sketch to your Arduino. When the upload is complete, check the Serial Monitor. You will see the class names printed next to the confidence in each class.

![Serial Monitor](GettingStartedImages/serial_mon.png)


