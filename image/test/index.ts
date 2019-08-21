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
// @ts-ignore
var Table = require("cli-table");

const SEED_WORD = "testSuite";
const seed: seedrandom.prng = seedrandom(SEED_WORD);
const DATASET_URL =
	"https://storage.googleapis.com/teachable-machine-models/test_data/image/flowers_all/";

/**
 * Load a flower image from our storage bucket
 */
function loadFlowerImage(c: string, i: number): Promise<HTMLImageElement> {
	// tslint:disable-next-line:max-line-length
	const src = DATASET_URL + `${c}/${i}.jpg`;
	// console.log(src);
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
	loadFunction: Function,
	classes: string[],
	trainSize: number,
	testSize: number
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
			load.push(loadFunction(c, i));
		}
		trainAndValidationImages.push(await Promise.all(load));

		load = [];
		for (const i of testIndices) {
			load.push(loadFunction(c, i));
		}
		testImages.push(await Promise.all(load));
	}

	return {
		trainAndValidationImages: trainAndValidationImages,
		testImages: testImages
	};
}

/**
 * Output loss and accuracy results at the end of training
 * Also evaluate the test dataset
 */
function showMetrics(alpha: number, time: number, logs: tf.Logs[], testAccuracy?: number) {
	const lastEpoch = logs[logs.length - 1];

	const header = "α=" + alpha + ", t=" + (time/1000).toFixed(3) + "s";

	const table = new Table({
		head: [header, "Accuracy", "Loss"],
		colWidths: [18, 10, 10]
	});

	table.push(
		[ "Train", lastEpoch.acc.toFixed(3), lastEpoch.loss.toFixed(3) ],
		[ "Validation", lastEpoch.val_acc.toFixed(3), lastEpoch.val_loss.toFixed(3) ],
		[ "Test", testAccuracy.toFixed(3), '']
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
	showEpochResults: boolean = false
) {
	model.setLabels(classes);
	model.setSeed(SEED_WORD); // set a seed to shuffle predictably

	const logs: tf.Logs[] = [];
	let time: number = 0;

	await tf.nextFrame().then(async () => {
		let index = 0;
		for (const imgSet of trainAndValidationImages) {
			for (const img of imgSet) {
				await model.addExample(index, img);
			}
			index++;
		}
		const start = window.performance.now();
		await model.train(
			{
				denseUnits: 100,
				epochs,
				learningRate: 0.0001,
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
					logs.push(log);
				}
			}
		);
		const end = window.performance.now();
		time = end - start;
	});

	// Analyze the test set (model has not seen for training)
	let accuracy = 0;
	for (let i = 0; i < classes.length; i++) {
		const classImages = testImages[i];

		for (const image of classImages) {
			const scores = await model.predict(image, false);
			// compare the label
			if (scores[0].className === classes[i]) {
				accuracy++;
			}
		}
	}
	const testAccuracy = accuracy / (testSizePerClass * classes.length);

	showMetrics(alpha, time, logs, testAccuracy);

	return testAccuracy;
}

// Weird workaround...
tf.util.fetch = (a, b) => window.fetch(a, b);

describe("Module exports", () => {
	it("should contain ", () => {
		assert.typeOf(tm, "object", "tm should be an object");
		assert.typeOf(
			tm.getWebcam,
			"function",
			"tm.getWebcam should be a function"
		);
		assert.typeOf(tm.version, "string", "tm.version should be a string");
		assert.typeOf(tm.CustomMobileNet, "function");
		assert.typeOf(tm.TeachableMobileNet, "function");
		assert.typeOf(tm.load, "function");
		assert.typeOf(tm.loadFromFiles, "function");
		assert.equal(tm.IMAGE_SIZE, 224, "IMAGE_SIZE should be 224");
	});
});

describe("Train a custom model", () => {
	it("create a model", async () => {
		const teachableMobileNet = await tm.createTeachable({
			tfjsVersion: tf.version.tfjs
			// tmVersion: version
		});		

		assert.exists(teachableMobileNet);
	}).timeout(5000);

	it("Train flower dataset on mobilenet v2", async () => {
		// classes, samplesPerClass, url
		const metadata = await (await fetch(
			DATASET_URL + "metadata.json"
		)).json();

		// 1. Setup dataset parameters
		const classLabels = metadata.classes as string[];
		const TRAIN_VALIDATION_SIZE_PER_CLASS = 10; 
		const TEST_SIZE_PER_CLASS = Math.ceil(
			(TRAIN_VALIDATION_SIZE_PER_CLASS * 0.1) / 0.9
		);

		var table = new Table();
		table.push(
			{
				"train/validation size":
					TRAIN_VALIDATION_SIZE_PER_CLASS * classLabels.length
			},
			{ "test size": TEST_SIZE_PER_CLASS * classLabels.length }
		);
		console.log("\n" + table.toString());

		// 2. Create our datasets once
		const datasets = await createDatasets(
			loadFlowerImage,
			classLabels,
			TRAIN_VALIDATION_SIZE_PER_CLASS,
			TEST_SIZE_PER_CLASS
		);
		const trainAndValidationImages = datasets.trainAndValidationImages;
		const testImages = datasets.testImages;

		// NOTE: If testing time, test first model twice because it takes longer 
		// to train the very first time tf.js is training 
		const MOBILENET_VERSION = 1;
		// const VALID_ALPHAS = [0.25];
		const VALID_ALPHAS = [0.25, 0.5, 0.75, 1];
		// const VALID_ALPHAS = [0.4];
		const EPOCHS = 20;

		for (let a of VALID_ALPHAS) {
			const lineStart = "\n//====================================";
			const lineEnd = "====================================//\n\n";
			console.log(lineStart);
			// 3. Test data on the model
			const teachableMobileNetV2 = await tm.createTeachable(
				{ tfjsVersion: tf.version.tfjs },
				{ version: MOBILENET_VERSION, alpha: a }
			);

			
			const accuracyV2 = await testModel(
				teachableMobileNetV2,
				a,
				classLabels,
				trainAndValidationImages,
				testImages,
				TEST_SIZE_PER_CLASS,
				EPOCHS,
				false
			);

			// assert.isTrue(accuracyV2 > 0.7);
			console.log(lineEnd);
		}
	}).timeout(500000);
});
