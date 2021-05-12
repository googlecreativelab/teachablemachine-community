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

import { assert } from 'chai';

import * as tm from '../src/index';
import * as tf from '@tensorflow/tfjs';
import { PoseModelSettings } from '../src/custom-posenet';

const dataset_url = 
    "https://storage.googleapis.com/teachable-machine-models/test_data/pose/arms/";

// Weird workaround...
tf.util.fetch = (a, b) => window.fetch(a, b);

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

async function testMetadata() {
    let poseModel = await tm.createTeachable({
        tfjsVersion: tf.version.tfjs,
        tmVersion: tm.version,
        modelSettings: {}
    });		
    assert.exists(poseModel.getMetadata().modelSettings);

    poseModel = await tm.createTeachable({
        tfjsVersion: tf.version.tfjs,
        tmVersion: tm.version,
        modelSettings: {
            posenet: {}
        }
    });
    assert.exists((poseModel.getMetadata().modelSettings as PoseModelSettings).posenet);

    poseModel = await tm.createTeachable({
        tfjsVersion: tf.version.tfjs,
        tmVersion: tm.version,
        modelSettings: {
            posenet: {
                architecture: 'MobileNetV1',
                outputStride: 8,
                multiplier: 0.50
            }
        }
    });
    assert.equal((poseModel.getMetadata().modelSettings as PoseModelSettings).posenet.outputStride, 8);
    assert.equal((poseModel.getMetadata().modelSettings as PoseModelSettings).posenet.multiplier, 0.5);

    poseModel = await tm.createTeachable({
        tfjsVersion: tf.version.tfjs,
        tmVersion: tm.version,
        modelSettings: {
            randomExtraKey: 4
        }
    });		
    assert.exists((poseModel.getMetadata().modelSettings as PoseModelSettings).posenet.outputStride);

    return poseModel;
}

async function testPosenet(
    epochs: number,
	learningRate: number,
	showEpochResults: boolean = false,
	earlyStopEpoch: number = epochs
) {
    const poseModel = await tm.createTeachable({
        tfjsVersion: tf.version.tfjs,
        tmVersion: tm.version
    });		
    assert.exists(poseModel);
    
    const metadata = await (await fetch(
        dataset_url + "metadata.json"
    )).json();

    const classLabels = metadata.classes as string[];

    assert.equal(classLabels.length, 2);

    const trainAndValidationImages: any[][] = [];
    
    // const trainingSize = Math.min(...metadata.samplesPerClass);
    const trainingSize = 10;
    for (const c of classLabels) {
        const load: Array<Promise<any>> = [];
        for (let i = 0; i < trainingSize; i++) {
            const src = dataset_url + `${c}/${i}.png`;
            const l = new Promise((resolve, reject) => {
                const img = new Image();
                img.onload = () => resolve(img);
                img.onerror = reject;
                img.crossOrigin = "anonymous";
                img.src = src;
            }).then(img  => {
                return poseModel.estimatePose(img as HTMLImageElement);
            }).then( output => output.posenetOutput);
            
            load.push(l);
        }
        trainAndValidationImages.push(await Promise.all(load));
    }

    assert.equal( trainAndValidationImages[0].length, trainingSize);    
    
    poseModel.setLabels(metadata.classes);
    poseModel.setSeed('testSuite'); // set a seed to shuffle predictably

    const logs: tf.Logs[] = [];

    await tf.nextFrame().then(async () => {
        let index = 0;
        for (const imgSet of trainAndValidationImages) {
            for (const img of imgSet) {
                await poseModel.addExample(index, img);
            }
            index++;
        }
        await poseModel.train(
            {
                denseUnits: 100,
                epochs: epochs,
                learningRate: learningRate,
                batchSize: 16
            },
            {
                onEpochBegin: async (epoch: number, logs: tf.Logs) => {
                    if (showEpochResults) {
                        console.log("Epoch: ", epoch);
                    }
                },
                onEpochEnd: async (epoch: number, log: tf.Logs) => {
                    logs.push(log);

                    if (earlyStopEpoch !== epochs && earlyStopEpoch === epoch) {
						poseModel.stopTraining().then(() => {
							console.log("Stopped training early");
						});
					}
                }
            }
        ); 
    });

    const lastLog = logs[logs.length-1];

    return { model: poseModel, lastEpoch: lastLog }
}

describe('Test pose library', () => {
    it('constants are set correctly', () => {
        assert.equal(typeof tm, 'object', 'tm should be an object');
        assert.equal(typeof tm.Webcam, 'function', 'tm.Webcam should be a function');
        assert.equal(typeof tm.version, 'string', 'tm.version should be a string');
        assert.equal(tm.version, require('../package.json').version, "version does not match package.json.");
    });

    it('metadata loads correctly', async () => {
        const model = await testMetadata();
        assert.exists(model);
    }).timeout(100000);;

    let poseModel: tm.TeachablePoseNet;

    it('can train pose model', async () => {
        const { model, lastEpoch } = await testPosenet(10, 0.0001, false);
        poseModel = model;
        assert.isAbove(lastEpoch.acc, 0.9);
        assert.isBelow(lastEpoch.loss, 0.001);
    }).timeout(1000000);

    it('test early stop', async () => {
        const { model, lastEpoch } = await testPosenet(10, 0.0001, false, 5);
        assert.isAbove(lastEpoch.acc, 0.9);
        assert.isBelow(lastEpoch.loss, 0.1);
    }).timeout(1000000);

    it("Test predict functions", async () => {
        let testImage, prediction, poseResult, predictionTopK;

        // test image 1
        testImage = await loadPngImage('arms', 0, dataset_url);
        poseResult = await poseModel.estimatePose(testImage, false);

        prediction = await poseModel.predict(poseResult.posenetOutput);
		assert.isAbove(prediction[0].probability, 0.9);

        predictionTopK = await poseModel.predictTopK(poseResult.posenetOutput, 3);
        assert.equal(predictionTopK[0].className, 'arms');
        assert.isAbove(predictionTopK[0].probability, 0.9);
        
         // test image 2
         testImage = await loadPngImage('no_arms', 0, dataset_url);
         poseResult = await poseModel.estimatePose(testImage, false);
 
         prediction = await poseModel.predict(poseResult.posenetOutput);
         assert.isAbove(prediction[1].probability, 0.9);
 
         predictionTopK = await poseModel.predictTopK(poseResult.posenetOutput, 3);
         assert.equal(predictionTopK[0].className, 'no_arms');
         assert.isAbove(predictionTopK[0].probability, 0.9);
	}).timeout(500000);

});
