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

type Drawable = HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | ImageBitmap;

const newCanvas = () => document.createElement('canvas');

export function resize(image: Drawable, scale: number, canvas: HTMLCanvasElement= newCanvas()) {
    canvas.width = image.width * scale;
    canvas.height = image.height * scale;
    const ctx: CanvasRenderingContext2D = canvas.getContext('2d')!;

    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas;
}

export function resizeMaxTo(image: Drawable, maxSize: number, canvas: HTMLCanvasElement= newCanvas()) {
    const max = Math.max(image.width, image.height);
    return resize(image, maxSize / max, canvas);
}

export function resizeMinTo(image: Drawable, minSize: number, canvas: HTMLCanvasElement= newCanvas()) {
    const min = Math.min(image.width, image.height);
    return resize(image, minSize / min, canvas);
}


export function cropTo( image: Drawable, size: number,
    flipped = false, canvas: HTMLCanvasElement = newCanvas()) {

    // image image, bitmap, or canvas
    let width = image.width;
    let height = image.height;

    // if video element
    if (image instanceof HTMLVideoElement) {
        width = (image as HTMLVideoElement).videoWidth;
        height = (image as HTMLVideoElement).videoHeight;
    }

    const min = Math.min(width, height);
    const scale = size / min;
    const scaledW = Math.ceil(width * scale);
    const scaledH = Math.ceil(height * scale);
    const dx = scaledW - size;
    const dy = scaledH - size;
    canvas.width = canvas.height = size;
    const ctx: CanvasRenderingContext2D = canvas.getContext('2d');
    ctx.drawImage(image, ~~(dx / 2) * -1, ~~(dy / 2) * -1, scaledW, scaledH);

    // canvas is already sized and cropped to center correctly
    if (flipped) {
        ctx.scale(-1, 1);
        ctx.drawImage(canvas, size * -1, 0);
    }

    return canvas;
}
