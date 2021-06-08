/* Copyright 2021 Google LLC. Bose All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
  ==============================================================================
 */

import processing.serial.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import websockets.*;
import javax.xml.bind.DatatypeConverter;
import controlP5.*;
import java.util.*;

Serial myPort;
WebsocketServer ws;

// must match resolution used in the sketch
final int cameraWidth = 96;
final int cameraHeight = 96;
final int cameraBytesPerPixel = 1;
final int bytesPerFrame = cameraWidth * cameraHeight * cameraBytesPerPixel;

PImage myImage;
byte[] frameBuffer = new byte[bytesPerFrame];
String[] portNames;
ControlP5 cp5;
ScrollableList portsList;
boolean clientConnected = false;

void setup()
{
  size(448, 224);
  pixelDensity(displayDensity());
  frameRate(30);
  cp5 = new ControlP5(this);
  portNames = Serial.list();
  portNames = filteredPorts(portNames);
  ws = new WebsocketServer(this, 8889, "/");
  portsList = cp5.addScrollableList("portSelect")
    .setPosition(235, 10)
    .setSize(200, 220)
    .setBarHeight(40)
    .setItemHeight(40)
    .addItems(portNames);

  portsList.close();
  // wait for full frame of bytes
  //myPort.buffer(bytesPerFrame);   
  //myPort = new Serial(this, "COM5", 9600);
  //myPort = new Serial(this, "/dev/ttyACM0", 9600);
  //myPort = new Serial(this, "/dev/cu.usbmodem14201", 9600);   

  myImage = createImage(cameraWidth, cameraHeight, RGB);
  noStroke();
}

void draw()
{  
  background(240);
  image(myImage, 0, 0, 224, 224);

  drawConnectionStatus();
}

void drawConnectionStatus() {
  fill(0);
  textAlign(RIGHT, CENTER);
  if (!clientConnected) {
    text("Not Connected to TM", 410, 100);
    fill(255, 0, 0);
  } else {
    text("Connected to TM", 410, 100);
    fill(0, 255, 0);
  }
  ellipse(430, 102, 10, 10);
}
void portSelect(int n) {
  String selectedPortName = (String) cp5.get(ScrollableList.class, "portSelect").getItem(n).get("text");

  try {
    myPort = new Serial(this, selectedPortName, 9600);
    myPort.buffer(bytesPerFrame);
  } 
  catch (Exception e) {
    println(e);
  }
}


boolean stringFilter(String s) {
  return (!s.startsWith("/dev/tty"));
}
int lastFrame = -1;
String [] filteredPorts(String[] ports) {
  int n = 0;
  for (String portName : ports) if (stringFilter(portName)) n++;
  String[] retArray = new String[n];
  n = 0;
  for (String portName : ports) if (stringFilter(portName)) retArray[n++] = portName; 
  return retArray;
}

void serialEvent(Serial myPort) {
  // read the saw bytes in
  myPort.readBytes(frameBuffer);
  //println(frameBuffer);

  // access raw bytes via byte buffer
  ByteBuffer bb = ByteBuffer.wrap(frameBuffer);
  bb.order(ByteOrder.BIG_ENDIAN);
  int i = 0;
  
  while (bb.hasRemaining()) { 
    //0xFF & to treat byte as unsigned.
    int r = (int) (bb.get() & 0xFF);
    myImage.pixels[i] = color(r, r, r);
    i++;
    //println("adding pixels");
  }
  if (lastFrame == -1) {
    lastFrame = millis();
  }
  else {
    int frameTime = millis() - lastFrame;
    print("fps: ");
    println(frameTime);
    lastFrame = millis();
  }
  
  myImage.updatePixels();
  myPort.clear();
  String data = DatatypeConverter.printBase64Binary(frameBuffer);
  ws.sendMessage(data);
}

void webSocketServerEvent(String msg) {
  if (msg.equals("tm-connected")) clientConnected = true;
}
