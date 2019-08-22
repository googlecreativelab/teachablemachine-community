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

import { Keypoint, Vector2D } from "@tensorflow-models/posenet/dist/types";
import { getAdjacentKeyPoints } from "@tensorflow-models/posenet/dist/util";

const FILL_COLOR = "aqua";
const STROKE_COLOR = "aqua";
const KEYPOINT_SIZE = 4;
const LINE_WIDTH = 2;

/**
 * Draw pose keypoints onto a canvas
 */
export function drawKeypoints(
	keypoints: Keypoint[],
	minConfidence: number,
	ctx: CanvasRenderingContext2D,
	keypointSize: number = KEYPOINT_SIZE,
	fillColor: string = FILL_COLOR,
	strokeColor: string = STROKE_COLOR,
	scale = 1
) {
	for (let i = 0; i < keypoints.length; i++) {
		const keypoint = keypoints[i];

		if (keypoint.score < minConfidence) {
			continue;
		}

		const { y, x } = keypoint.position;

		drawPoint(
			ctx,
			y * scale,
			x * scale,
			keypointSize,
			fillColor,
			strokeColor
		);
	}
}

export function drawPoint(
	ctx: CanvasRenderingContext2D,
	y: number,
	x: number,
	keypointSize: number,
	fillColor: string,
	strokeColor: string
) {
	ctx.fillStyle = fillColor;
	ctx.strokeStyle = strokeColor;
	ctx.beginPath();
	ctx.arc(x, y, keypointSize, 0, 2 * Math.PI);
	ctx.fill();
	ctx.stroke();
}

export function toTuple(position: Vector2D) {
	return [position.y, position.x];
}

export function drawSkeleton(
	keypoints: Keypoint[],
	minConfidence: number,
	ctx: CanvasRenderingContext2D,
	lineWidth: number = LINE_WIDTH,
	strokeColor: string = STROKE_COLOR,
	scale = 1
) {
	const adjacentKeyPoints = getAdjacentKeyPoints(keypoints, minConfidence);

	adjacentKeyPoints.forEach(keypoints => {
		drawSegment(
			toTuple(keypoints[0].position),
			toTuple(keypoints[1].position),
			ctx,
			lineWidth,
			strokeColor,
			scale
		);
	});
}

export function drawSegment(
	[ay, ax]: number[],
	[by, bx]: number[],
	ctx: CanvasRenderingContext2D,
	lineWidth: number,
	strokeColor: string,
	scale: number
) {
	ctx.beginPath();
	ctx.moveTo(ax * scale, ay * scale);
	ctx.lineTo(bx * scale, by * scale);
	ctx.lineWidth = lineWidth;
	ctx.strokeStyle = strokeColor;
	ctx.stroke();
}
