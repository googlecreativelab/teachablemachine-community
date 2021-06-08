/* Copyright 2021 Google LLC All Rights Reserved.

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


#include "ImageProvider.h"


/*
  OV767X - Camera Capture Raw Bytes

  This sketch reads a frame from the OmniVision OV7670 camera
  and writes the bytes to the Serial port. Use the Processing
  sketch in the extras folder to visualize the camera output.

  Circuit:
    - Arduino Nano 33 BLE board
    - OV7670 camera module:
      - 3.3 connected to 3.3
      - GND connected GND
      - SIOC connected to A5
      - SIOD connected to A4
      - VSYNC connected to 8
      - HREF connected to A1
      - PCLK connected to A0
      - XCLK connected to 9
      - D7 connected to 4
      - D6 connected to 6
      - D5 connected to 5
      - D4 connected to 3
      - D3 connected to 2
      - D2 connected to 0 / RX
      - D1 connected to 1 / TX
      - D0 connected to 10

  This example code is in the public domain.
*/
#include <Arduino.h>
#include <Arduino_OV767X.h>

const int kCaptureWidth = 320;
const int kCaptureHeight = 240;
const int capDataLen = kCaptureWidth * kCaptureHeight * 2;
byte captured_data[kCaptureWidth * kCaptureHeight * 2]; // QVGA: 320x240 X 2 bytes per pixel (RGB565)


// Crop image and convert it to grayscale
boolean ProcessImage(
  int image_width, int image_height,
  uint8_t* image_data) {
  //  Serial.println("begging image process");
  //  const int skip_start_x = ceil((kCaptureWidth - image_width) / 2); // 40.5
  //  const int skip_start_y = ceil((kCaptureHeight - image_height) / 2); // 24
  //  const int skip_end_x_index = (kCaptureWidth - skip_start_x); // (176 - 40) = 135
  //  const int skip_end_y_index = (kCaptureHeight - skip_start_y); // 144 - 24 = 120
  const int imgSize = 96;

  // Color of the current pixel
  uint16_t color;
  for (int y = 0; y < imgSize; y++) {
    for (int x = 0; x < imgSize; x++) {
      int currentCapX = floor(map(x, 0, imgSize, 40, kCaptureWidth - 80));
      int currentCapY = floor(map(y, 0, imgSize, 0, kCaptureHeight));
      // Read the color of the pixel as 16-bit integer
      int read_index = (currentCapY * kCaptureWidth + currentCapX) * 2;//(y * kCaptureWidth + x) * 2;
      int i2 = (currentCapY * kCaptureWidth + currentCapX + 1) * 2;
      int i3 = ((currentCapY + 1) * kCaptureWidth + currentCapX) * 2;
      int i4 = ((currentCapY + 1) * kCaptureWidth + currentCapX + 1) * 2;

      uint8_t high_byte = captured_data[read_index];
      uint8_t low_byte = captured_data[read_index + 1];

      color = ((uint16_t)high_byte << 8) | low_byte;
      // Extract the color values (5 red bits, 6 green, 5 blue)
      uint8_t r, g, b;
      r = ((color & 0xF800) >> 11) * 8;
      g = ((color & 0x07E0) >> 5) * 4;
      b = ((color & 0x001F) >> 0) * 8;
      // Convert to grayscale by calculating luminance
      // See https://en.wikipedia.org/wiki/Grayscale for magic numbers
      float gray_value = (0.2126 * r) + (0.7152 * g) + (0.0722 * b);

      if (i2 > 0 && i2 < capDataLen - 1) {
        high_byte = captured_data[i2];
        low_byte = captured_data[i2 + 1];
        color = ((uint16_t)high_byte << 8) | low_byte;
        r = ((color & 0xF800) >> 11) * 8;
        g = ((color & 0x07E0) >> 5) * 4;
        b = ((color & 0x001F) >> 0) * 8;
        gray_value += (0.2126 * r) + (0.7152 * g) + (0.0722 * b);
      }
      if (i3 > 0 && i3 < capDataLen - 1) {
        high_byte = captured_data[i3];
        low_byte = captured_data[i3 + 1];
        color = ((uint16_t)high_byte << 8) | low_byte;
        r = ((color & 0xF800) >> 11) * 8;
        g = ((color & 0x07E0) >> 5) * 4;
        b = ((color & 0x001F) >> 0) * 8;
        gray_value += (0.2126 * r) + (0.7152 * g) + (0.0722 * b);
      }

      if (i4 > 0 && i4 < capDataLen - 1) {
        high_byte = captured_data[i4];
        low_byte = captured_data[i4 + 1];
        color = ((uint16_t)high_byte << 8) | low_byte;
        r = ((color & 0xF800) >> 11) * 8;
        g = ((color & 0x07E0) >> 5) * 4;
        b = ((color & 0x001F) >> 0) * 8;
        gray_value += (0.2126 * r) + (0.7152 * g) + (0.0722 * b);
      }

      gray_value = gray_value / 4;

      // Convert to signed 8-bit integer by subtracting 128.
      //
      //      // The index of this pixel` in our flat output buffer
      int index = y * image_width + x;
      image_data[index] = static_cast<int8_t>(gray_value);
//      delayMicroseconds(10);
    }
  }
//  flushCap();
  //  Serial.println("processed image");
  return true;
}

// Get an image from the camera module
boolean GetImage(int image_width,
                 int image_height, int channels, uint8_t* image_data) {
  static bool g_is_camera_initialized = false;

  if (!g_is_camera_initialized) {
    if (!Camera.begin(QVGA, RGB565, 1)) {

      return false;
    }
    g_is_camera_initialized = true;
  }


  Camera.readFrame(captured_data);

  boolean process_status = ProcessImage(image_width, image_height, image_data);
  if ( process_status != true) {
    //    Serial.println("process failed");
    return process_status;
  }

  return false;
}
