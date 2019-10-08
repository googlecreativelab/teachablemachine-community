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

const dataset_url = 
    "https://storage.googleapis.com/teachable-machine-models/test_data/pose/arms/";

// Weird workaround...
tf.util.fetch = (a, b) => window.fetch(a, b);


describe('Test pose library', () => {
    it('constants are set correctly', () => {
        assert.equal(typeof tm, 'object', 'tm should be an object');
        assert.equal(typeof tm.getWebcam, 'function', 'tm.getWebcam should be a function');
        assert.equal(typeof tm.version, 'string', 'tm.version should be a string');
        assert.equal(tm.version, require('../package.json').version, "version does not match package.json.");
    });

    it('can train pose model', async () => {
        const pose = await tm.createTeachable({
            tfjsVersion: tf.version.tfjs,
            tmVersion: tm.version
		});		
        assert.exists(pose);
        
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
                // console.log(src)
                const l = new Promise((resolve, reject) => {
                    const img = new Image();
                    img.onload = () => resolve(img);
                    img.onerror = reject;
                    img.crossOrigin = "anonymous";
                    img.src = src;
                }).then(img  => {
                    return pose.estimatePose(img as HTMLImageElement);
                }).then( output => output.posenetOutput);
                
                load.push(l);
            }
            trainAndValidationImages.push(await Promise.all(load));
        }

      
        assert.equal( trainAndValidationImages[0].length, trainingSize);

        let EPOCHS = 10;
        let LEARNING_RATE = 0.0001;        
        
        pose.setLabels(metadata.classes);
        pose.setSeed('testSuite'); // set a seed to shuffle predictably
    
        const logs: tf.Logs[] = [];
    
        await tf.nextFrame().then(async () => {
            let index = 0;
            for (const imgSet of trainAndValidationImages) {
                for (const img of imgSet) {
                    await pose.addExample(index, img);
                }
                index++;
            }
            await pose.train(
                {
                    denseUnits: 100,
                    epochs: EPOCHS,
                    learningRate: LEARNING_RATE,
                    batchSize: 16
                },
                {
                    onEpochBegin: async (epoch: number, logs: tf.Logs) => {
                        // if (showEpochResults) {
                            console.log("Epoch: ", epoch);
                        // }
                    },
                    onEpochEnd: async (epoch: number, log: tf.Logs) => {
                        // console.log(log);
                        logs.push(log);
                    }
                }
            );
            
        });

        const lastLog = logs[logs.length-1];
        assert.isAbove(lastLog.acc, 0.9);
        assert.isBelow(lastLog.loss, 0.001);

    }).timeout(100000);

});
