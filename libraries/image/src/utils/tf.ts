/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

/**
 * Receives an image and normalizes it between -1 and 1.
 * Returns a batched image (1 - element batch) of shape [1, w, h, c]
 * @param rasterElement the element with pixels to convert to a Tensor
 * @param grayscale optinal flag that changes the crop to [1, w, h, 1]
 */
export function capture(rasterElement: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement, grayscale?: boolean) {
    return tf.tidy(() => {
        const pixels = tf.browser.fromPixels(rasterElement);

        // crop the image so we're using the center square
        const cropped = cropTensor(pixels, grayscale);

        // Expand the outer most dimension so we have a batch size of 1
        const batchedImage = cropped.expandDims(0);

        // Normalize the image between -1 and a1. The image comes in between 0-255
        // so we divide by 127 and subtract 1.
        return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    });
}


export function cropTensor( img: tf.Tensor3D, grayscaleModel?: boolean, grayscaleInput?: boolean ) : tf.Tensor3D {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    
    if (grayscaleModel && !grayscaleInput) {
        //cropped rgb data
        let grayscale_cropped = img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
        
        grayscale_cropped = grayscale_cropped.reshape([size * size, 1, 3])
        const rgb_weights = [0.2989, 0.5870, 0.1140]
        grayscale_cropped = tf.mul(grayscale_cropped, rgb_weights)
        grayscale_cropped = grayscale_cropped.reshape([size, size, 3]);
    
        grayscale_cropped = tf.sum(grayscale_cropped, -1)
        grayscale_cropped = tf.expandDims(grayscale_cropped, -1)

        return grayscale_cropped;
    }
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}
