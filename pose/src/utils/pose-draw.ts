import { Keypoint, Vector2D } from "@tensorflow-models/posenet/dist/types";
import { getAdjacentKeyPoints } from "@tensorflow-models/posenet/dist/util";

const FILL_COLOR = 'aqua';
const STROKE_COLOR = FILL_COLOR;
const KEYPOINT_SIZE = 4;
const LINE_WIDTH = 2;

/**
 * Draw pose keypoints onto a canvas
 */
export function drawKeypoints(keypoints: Keypoint[], minConfidence: number, ctx: CanvasRenderingContext2D, scale = 1) {
  for (let i = 0; i < keypoints.length; i++) {
    const keypoint = keypoints[i];

    if (keypoint.score < minConfidence) {
      continue;
    }

    const {y, x} = keypoint.position;

    drawPoint(ctx, y * scale, x * scale, KEYPOINT_SIZE);
  }
}

export function drawPoint(ctx: CanvasRenderingContext2D, y: number, x: number, r: number) {
    ctx.fillStyle = FILL_COLOR;
    ctx.strokeStyle = STROKE_COLOR;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
}

export function toTuple(position: Vector2D) {
    return [position.y, position.x];
}

export function drawSkeleton(keypoints: Keypoint[], minConfidence: number, ctx: CanvasRenderingContext2D, scale = 1) {
  const adjacentKeyPoints = getAdjacentKeyPoints(keypoints, minConfidence);

  adjacentKeyPoints.forEach((keypoints) => {
    drawSegment(toTuple(keypoints[0].position), toTuple(keypoints[1].position), scale, ctx);
  });
}

export function drawSegment([ay, ax]: number[], [by, bx]: number[], scale: number, ctx: CanvasRenderingContext2D) {
    ctx.beginPath();
    ctx.moveTo(ax * scale, ay * scale);
    ctx.lineTo(bx * scale, by * scale);
    ctx.lineWidth = LINE_WIDTH;
    ctx.strokeStyle = STROKE_COLOR;
    ctx.stroke();
}
  