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

import { assert } from "chai";
import "mocha";

import * as tf from "@tensorflow/tfjs";
import * as tm from "../src/index";
import * as seedrandom from "seedrandom";
import { TeachableMobileNet } from "../src/index";
import { assertTypesMatch } from "@tensorflow/tfjs-core/dist/tensor_util";
import { cropTo } from "../src/utils/canvas";

// @ts-ignore
var Table = require("cli-table");

const SEED_WORD = "testSuite";
const seed: seedrandom.prng = seedrandom(SEED_WORD);
const FLOWER_DATASET_URL =
	"https://storage.googleapis.com/teachable-machine-models/test_data/image/flowers_all/";
const ELMO_DATASET_URL =
	"https://storage.googleapis.com/teachable-machine-models/test_data/image/elmo/";
const BEAN_DATASET_URL =
	"https://storage.googleapis.com/teachable-machine-models/test_data/image/beans/";
const FACE_DATASET_URL =
	"https://storage.googleapis.com/teachable-machine-models/test_data/image/face/";
const PLANT_DATASET_URL =
	"https://storage.googleapis.com/teachable-machine-models/test_data/image/plants/"

/**
 * Load a flower image from our storage bucket
 */
function loadJpgImage(c: string, i: number, dataset_url: string): Promise<HTMLImageElement> {
	// tslint:disable-next-line:max-line-length
	const src = dataset_url + `${c}/${i}.jpg`;
	return new Promise((resolve, reject) => {
		const img = new Image();
		img.onload = () => resolve(img);
		img.onerror = reject;
		img.crossOrigin = "anonymous";
		img.src = src;
	});
}


function loadPngImage(c: string, i: number, dataset_url: string): Promise<HTMLImageElement> {
	// tslint:disable-next-line:max-line-length
	const src = dataset_url + `${c}/${i}.png`;

	// console.log(src)
	return new Promise((resolve, reject) => {
		const img = new Image();
		img.onload = () => resolve(img);
		img.onerror = reject;
		img.crossOrigin = "anonymous";
		img.src = src;
	});
}

/**
 * Shuffle an array of Float32Array or Samples using Fisher-Yates algorithm
 * Takes an optional seed value to make shuffling predictable
 */
function fisherYates(array: number[], seed?: seedrandom.prng) {
	const length = array.length;
	const shuffled = array.slice(0);
	for (let i = length - 1; i > 0; i -= 1) {
		let randomIndex;
		if (seed) {
			randomIndex = Math.floor(seed() * (i + 1));
		} else {
			randomIndex = Math.floor(Math.random() * (i + 1));
		}
		[shuffled[i], shuffled[randomIndex]] = [
			shuffled[randomIndex],
			shuffled[i]
		];
	}
	return shuffled;
}

/**
 * Create train/validation dataset and test dataset with unique images
 */
async function createDatasets(
	dataset_url: string,
	classes: string[],
	trainSize: number,
	testSize: number,
	loadFunction: Function
) {
	// fill in an array with unique numbers
	let listNumbers = [];
	for (let i = 0; i < trainSize + testSize; ++i) listNumbers[i] = i;
	listNumbers = fisherYates(listNumbers, seed); // shuffle

	const trainAndValidationIndeces = listNumbers.slice(0, trainSize);
	const testIndices = listNumbers.slice(trainSize, trainSize + testSize);

	const trainAndValidationImages: HTMLImageElement[][] = [];
	const testImages: HTMLImageElement[][] = [];

	for (const c of classes) {
		let load: Array<Promise<HTMLImageElement>> = [];
		for (const i of trainAndValidationIndeces) {
			load.push(loadFunction(c, i, dataset_url));
		}
		trainAndValidationImages.push(await Promise.all(load));

		load = [];
		for (const i of testIndices) {
			load.push(loadFunction(c, i, dataset_url));
		}
		testImages.push(await Promise.all(load));
	}

	return {
		trainAndValidationImages,
		testImages
	};
}

/**
 * Output loss and accuracy results at the end of training
 * Also evaluate the test dataset
 */
function showMetrics(alpha: number, time: number, logs: tf.Logs[], testAccuracy?: number) {
	const lastEpoch = logs[logs.length - 1];

	const header = "α=" + alpha + ", t=" + (time/1000).toFixed(1) + "s";

	const table = new Table({
		head: [header, "Accuracy", "Loss"],
		colWidths: [18, 10, 10]
	});

	table.push(
		[ "Train", lastEpoch.acc.toFixed(3), lastEpoch.loss.toFixed(5) ],
		[ "Validation", lastEpoch.val_acc.toFixed(3), lastEpoch.val_loss.toFixed(5) ]
	);
	console.log("\n" + table.toString());
}

async function testModel(
	model: TeachableMobileNet,
	alpha: number,
	classes: string[],
	trainAndValidationImages: HTMLImageElement[][],
	testImages: HTMLImageElement[][],
	testSizePerClass: number,
	epochs: number,
	learningRate: number,
	showEpochResults: boolean = false,
	earlyStopEpoch: number = epochs,
	imageSize?: number,
) {
	model.setLabels(classes);
	model.setSeed(SEED_WORD); // set a seed to shuffle predictably

	const logs: tf.Logs[] = [];
	let time: number = 0;

	await tf.nextFrame().then(async () => {
		let index = 0;
		for (const imgSet of trainAndValidationImages) {
			for (const img of imgSet) {
				if (imageSize) {
					let croppedImg = cropTo(img, 96, false);
					await model.addExample(index, croppedImg);
				}
				else {
					await model.addExample(index, img);
				}
			}
			index++;
		}
		const start = window.performance.now();
		await model.train(
			{
				denseUnits: 100,
				epochs,
				learningRate,
				batchSize: 16
			},
			{
				onEpochBegin: async (epoch: number, logs: tf.Logs) => {
					if (showEpochResults) {
						console.log("Epoch: ", epoch);
					}
				},
				onEpochEnd: async (epoch: number, log: tf.Logs) => {
					if (showEpochResults) {
						console.log(log);
					}
					if (earlyStopEpoch !== epochs && earlyStopEpoch === epoch) {
						model.stopTraining().then(() => {
							console.log("Stopped training early");
						});
					}
					logs.push(log);
				}
			}
		);
		const end = window.performance.now();
		time = end - start;
	});

	// // Analyze the test set (model has not seen for training)
	// let accuracy = 0;
	// for (let i = 0; i < classes.length; i++) {
	// 	const classImages = testImages[i];

	// 	for (const image of classImages) {
	// 		const scores = await model.predict(image, false);
	// 		// compare the label
	// 		if (scores[0].className === classes[i]) {
	// 			accuracy++;
	// 		}
	// 	}
	// }
	// const testAccuracy = accuracy / (testSizePerClass * classes.length);

	showMetrics(alpha, time, logs);
	return logs[logs.length - 1];


}

async function testMobilenet(dataset_url: string, version: number, loadFunction: Function, maxImages: number = 200, earlyStop: boolean = false, grayscale: boolean = false){
	// classes, samplesPerClass, url
	const metadata = await (await fetch(
		dataset_url + "metadata.json"
	)).json();
	// 1. Setup dataset parameters
	const classLabels = metadata.classes as string[];

	let NUM_IMAGE_PER_CLASS = Math.ceil(maxImages / classLabels.length); 

	if(NUM_IMAGE_PER_CLASS > Math.min(...metadata.samplesPerClass)){
		NUM_IMAGE_PER_CLASS = Math.min(...metadata.samplesPerClass);
	}
	const TRAIN_VALIDATION_SIZE_PER_CLASS =  NUM_IMAGE_PER_CLASS

	const table = new Table();
	table.push(
		{
			"train/validation size":
				TRAIN_VALIDATION_SIZE_PER_CLASS * classLabels.length
		}
	);
	console.log("\n" + table.toString());

	// 2. Create our datasets once
	const datasets = await createDatasets(
		dataset_url,
		classLabels,
		TRAIN_VALIDATION_SIZE_PER_CLASS,
		0,
		loadFunction
	);
	const trainAndValidationImages = datasets.trainAndValidationImages;
	const testImages = datasets.testImages;

	// NOTE: If testing time, test first model twice because it takes longer 
	// to train the very first time tf.js is training 


	const MOBILENET_VERSION = version;
	let VALID_ALPHAS = [0.35];
	// const VALID_ALPHAS = [0.25, 0.5, 0.75, 1];
	// const VALID_ALPHAS = [0.4];
	let EPOCHS = 50;
	let LEARNING_RATE = 0.001;
	if(version === 1){
		LEARNING_RATE = 0.0001;
		VALID_ALPHAS = [0.25];
		EPOCHS = 20;
	}

	const earlyStopEpochs = earlyStop ? 5 : EPOCHS;

	for (let a of VALID_ALPHAS) {
		const lineStart = "\n//====================================";
		const lineEnd = "====================================//\n\n";
		console.log(lineStart);
		// 3. Test data on the model
		let teachableMobileNet;
		let imageSize;
		if (grayscale) {
			imageSize = 96;
			teachableMobileNet = await tm.createTeachable(
				{ tfjsVersion: tf.version.tfjs, grayscale: true, imageSize },
				{ version: 1, alpha: 0.25, checkpointUrl: 'https://storage.googleapis.com/teachable-machine-models/mobilenet_v1_grayscale_025_96/model.json',
				trainingLayer: 'conv_pw_13_relu'}
			);
		}
		else {
			teachableMobileNet = await tm.createTeachable(
				{ tfjsVersion: tf.version.tfjs },
				{ version: MOBILENET_VERSION, alpha: a }
			);
		}
		
		const lastEpoch = await testModel(
			teachableMobileNet,
			a,
			classLabels,
			trainAndValidationImages,
			testImages,
			0,
			EPOCHS,
			LEARNING_RATE,
			false,
			earlyStopEpochs,
			imageSize
		);
			
		// assert.isTrue(accuracyV2 > 0.7);
		console.log(lineEnd);

		return { model: teachableMobileNet, lastEpoch };
	}
}

// Weird workaround...
tf.util.fetch = (a, b) => window.fetch(a, b);

describe("Module exports", () => {
	it("should contain ", () => {
		assert.typeOf(tm, "object", "tm should be an object");
		assert.typeOf(tm.Webcam, "function");
		assert.typeOf(tm.version, "string", "tm.version should be a string");
		assert.typeOf(tm.CustomMobileNet, "function");
		assert.typeOf(tm.TeachableMobileNet, "function");
		assert.typeOf(tm.load, "function");
		assert.typeOf(tm.loadFromFiles, "function");
		assert.equal(tm.IMAGE_SIZE, 224, "IMAGE_SIZE should be 224");
		// tslint:disable-next-line: no-require-imports
		assert.equal(tm.version, require('../package.json').version, "version does not match package.json.");
	});
});

describe("CI Test", () => {
	it("create a model", async () => {
		const teachableMobileNet = await tm.createTeachable(
			{ tfjsVersion: tf.version.tfjs },
			{ version: 2 }
		);
		assert.exists(teachableMobileNet);
	}).timeout(5000);

	let testModel: tm.TeachableMobileNet;

	it("Test tiny model (for CI)", async () => {
		const { model, lastEpoch } = await testMobilenet(BEAN_DATASET_URL, 2, loadPngImage, 10);
		testModel = model;
		assert.isAbove(lastEpoch.val_acc, 0.8);
		assert.isBelow(lastEpoch.val_loss, 0.1);
	}).timeout(500000);

	it("Test early stop", async () => {
		const { model, lastEpoch } = await testMobilenet(BEAN_DATASET_URL, 2, loadPngImage, 10, true);
		assert.isAbove(lastEpoch.val_acc, 0.8);
		assert.isBelow(lastEpoch.val_loss, 0.1);
	}).timeout(500000);

	it("Test predict functions", async () => {
		let testImage, prediction, predictionTopK;

		testImage = await loadPngImage('bad_bean', 0, BEAN_DATASET_URL);
		prediction = await testModel.predict(testImage, false);
		assert.isAbove(prediction[1].probability, 0.9);
		predictionTopK = await testModel.predictTopK(testImage, 3, false);
		assert.equal(predictionTopK[0].className, 'bad_bean');
		assert.isAbove(predictionTopK[0].probability, 0.9);

		testImage = await loadPngImage('good_bean', 0, BEAN_DATASET_URL);
		prediction = await testModel.predict(testImage, false);
		assert.isAbove(prediction[0].probability, 0.9);
		predictionTopK = await testModel.predictTopK(testImage, 3, false);
		assert.equal(predictionTopK[0].className, 'good_bean');
		assert.isAbove(predictionTopK[0].probability, 0.9);
	}).timeout(500000);


	it("creates a grayscale model", async() => {
		const grayscaleMobilenet = await tm.createTeachable(
			{ tfjsVersion: tf.version.tfjs, grayscale: true, imageSize: 96 },
			{ version: 1, alpha: 0.25, checkpointUrl: 'https://storage.googleapis.com/teachable-machine-models/mobilenet_v1_grayscale_025_96/model.json',
			trainingLayer: 'conv_pw_13_relu'}
		)

		assert.exists(grayscaleMobilenet);
	}).timeout(5000);

	let testGrayscaleModel: tm.TeachableMobileNet;
	it('test grayscale model accuracy (for CI)', async () => {
		const { model, lastEpoch } = await testMobilenet(PLANT_DATASET_URL, 1, loadJpgImage, 30, false, true);
		testGrayscaleModel = model;
		assert.isAbove(lastEpoch.val_acc, 0.8);
		assert.isBelow(lastEpoch.val_loss, 0.2);
		
	}).timeout(50000);

	it('tests graysale predict functions', async() => {
		let testImage, prediction, predictionTopK;
		testImage = await loadJpgImage('ficus', 0, PLANT_DATASET_URL);
		prediction = await testGrayscaleModel.predict(testImage, false);
		assert.isAbove(prediction[1].probability, 0.8);
		predictionTopK = await testGrayscaleModel.predictTopK(testImage, 3, false);
		assert.equal(predictionTopK[0].className, 'ficus');
		assert.isAbove(predictionTopK[0].probability, 0.8);

		testImage = await loadJpgImage('lily', 0, PLANT_DATASET_URL);
		prediction = await testGrayscaleModel.predict(testImage, false);
		assert.isAbove(prediction[0].probability, 0.8);
		predictionTopK = await testGrayscaleModel.predictTopK(testImage, 3, false);
		assert.equal(predictionTopK[0].className, 'lily');
		assert.isAbove(predictionTopK[0].probability, 0.8);
	})
});

// These test compare multiple models. Needs to run in non-headless chrome
describe.skip("Performance test", () => {
	it("Train flower dataset on mobilenet v1", async () => {
		console.log("Flower dataset mobilenet v1");
		await testMobilenet(FLOWER_DATASET_URL, 1, loadJpgImage);
	}).timeout(500000);

	it("Train flower dataset on mobilenet v2", async () => {
		console.log("Flower dataset mobilenet v2");
		await testMobilenet(FLOWER_DATASET_URL, 2, loadJpgImage);
	}).timeout(500000);

	it("Train elmo dataset on mobilenet v1", async () => {
		console.log("Elmo dataset mobilenet v1");
		await testMobilenet(ELMO_DATASET_URL, 1, loadPngImage);
	}).timeout(500000);

	it("Train elmo dataset on mobilenet v2", async () => {
		console.log("Elmo dataset mobilenet v2");
		await testMobilenet(ELMO_DATASET_URL, 2, loadPngImage);
	}).timeout(500000);

	it("Train bean dataset on mobilenet v1", async () => {
		console.log("Bean dataset mobilenet v1");
		await testMobilenet(BEAN_DATASET_URL, 1, loadPngImage);
	}).timeout(500000);

	it("Train bean dataset on mobilenet v2", async () => {
		console.log("Bean dataset mobilenet v2");
		await testMobilenet(BEAN_DATASET_URL, 2, loadPngImage);
	}).timeout(500000);

	it("Train face dataset on mobilenet v1", async () => {
		console.log("Face dataset mobilenet v1");
		await testMobilenet(FACE_DATASET_URL, 1, loadPngImage);
	}).timeout(500000);

	it("Train face dataset on mobilenet v2", async () => {
		console.log("Face dataset mobilenet v2");
		await testMobilenet(FACE_DATASET_URL, 2, loadPngImage);
	}).timeout(500000);


});
