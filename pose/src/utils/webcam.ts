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

const defaultVideoOptions: MediaTrackConstraints = {
    facingMode: 'user',
    frameRate: 30,
    aspectRatio: 1
};

/**
 * utility to get a webcam feed in a video element
 * @returns Promise<HTMLVideoElement>
 */
export function getWebcam(
    width = 400,
    height = 400,
    facingMode = 'user',
    flipped = true,
    video: HTMLVideoElement= document.createElement('video'),
    options: MediaTrackConstraints= defaultVideoOptions ) {
    if (!window.navigator.mediaDevices || !window.navigator.mediaDevices.getUserMedia) {
        return Promise.reject('Your browser does not support WebRTC. Please try another one.');
    }

    defaultVideoOptions.width = width;
    defaultVideoOptions.height = height;

    if (facingMode.toLowerCase() === 'back') {
        defaultVideoOptions.facingMode = 'environment';
    }

    // should be flipped since we trained with flipped webcam
    // but user can still change
    if (flipped) {
        const rotate = (value: string) =>
            `rotateY(${value}); -webkit-transform:rotateY(${value}); -moz-transform:rotateY(${value})`;
        video.setAttribute('style', `transform: ${rotate('180deg')}`);
    }

    return window.navigator.mediaDevices.getUserMedia({ video: options })
        .then((mediaStream) => {
            video.srcObject = mediaStream;
            video.width = width;
            video.height = height;
            return video;
        }, () => {
            return Promise.reject('Could not open your camera. You may have denied access.');
        });
}
