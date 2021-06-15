/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
  ==============================================================================*/

#include "image_provider.h"

/*
   The sample requires the following third-party libraries to be installed and
   configured:

   Arducam
   -------
   1. Download https://github.com/ArduCAM/Arduino and copy its `ArduCAM`
      subdirectory into `Arduino/libraries`. Commit #e216049 has been tested
      with this code.
   2. Edit `Arduino/libraries/ArduCAM/memorysaver.h` and ensure that
      "#define OV2640_MINI_2MP_PLUS" is not commented out. Ensure all other
      defines in the same section are commented out.

   JPEGDecoder
   -----------
   1. Install "JPEGDecoder" 1.8.0 from the Arduino library manager.
   2. Edit "Arduino/Libraries/JPEGDecoder/src/User_Config.h" and comment out
      "#define LOAD_SD_LIBRARY" and "#define LOAD_SDFAT_LIBRARY".
*/

#if defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)

#ifndef ARDUINO_EXCLUDE_CODE

#include <Arduino.h>
#include <Arduino_OV767X.h>

const int kCaptureWidth = 320;
const int kCaptureHeight = 240;

byte captured_data[kCaptureWidth * kCaptureHeight * 2]; // QVGA: 320x240 X 2 bytes per pixel (RGB565)

// Crop image and convert it to grayscale
TfLiteStatus ProcessImage(
  tflite::ErrorReporter* error_reporter,
  int image_width, int image_height,
  int8_t* image_data) {
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
      gray_value -= 128;
      //      // The index of this pixel` in our flat output buffer
      int index = y * image_width + x;
      image_data[index] = static_cast<int8_t>(gray_value);
//      delayMicroseconds(10);
    }
  }
//  flushCap();
  //  Serial.println("processed image");
  //  Serial.println("processed image");
  return kTfLiteOk;
}
// Get an image from the camera module
TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
                      int image_height, int channels, int8_t* image_data) {
  static bool g_is_camera_initialized = false;
  if (!g_is_camera_initialized) {
   if (!Camera.begin(QVGA, RGB565, 1)) {

      return kTfLiteError;
    }
    g_is_camera_initialized = true;
  }

    Camera.readFrame(captured_data);

  TfLiteStatus decode_status = ProcessImage(
                                 error_reporter, image_width, image_height, image_data);
  if (decode_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "DecodeAndProcessImage failed");
    return decode_status;
  }

  return kTfLiteOk;
}

#endif  // ARDUINO_EXCLUDE_CODE