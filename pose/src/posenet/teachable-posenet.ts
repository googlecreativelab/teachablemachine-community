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
import { PoseNet } from '@tensorflow-models/posenet'
import { util } from '@tensorflow/tfjs';
import { TensorContainer } from '@tensorflow/tfjs-core/dist/tensor_types';
import { CustomCallbackArgs, equalStrict } from '@tensorflow/tfjs';
import { CustomPoseNet, Metadata, loadPoseNet } from './custom-posenet';
import * as seedrandom from 'seedrandom';

const VALIDATION_FRACTION = 0.15;

export interface TrainingParameters {
    denseUnits: number;
    epochs: number;
    learningRate: number;
    batchSize: number;
}

interface Sample {
    data: Float32Array;
    label: number[];
}

// tslint:disable-next-line:no-any
const isTensor = (c: any): c is tf.Tensor =>
    typeof c.dataId === 'object' && c.shape === 'object';

/**
 * Converts an integer into its one-hot representation and returns
 * the data as a JS Array.
 */
function flatOneHot(label: number, numClasses: number) {
    const labelOneHot = new Array(numClasses).fill(0) as number[];
    labelOneHot[label] = 1;

    return labelOneHot;
}

/**
 * Shuffle an array of Float32Array or Samples using Fisher-Yates algorithm
 * Takes an optional seed value to make shuffling predictable
 */
function fisherYates(array: Float32Array[] | Sample[], seed?: seedrandom.prng) {
    const length = array.length;

    // need to clone array or we'd be editing original as we goo
    let shuffled = array.slice();

    for (let i = (length - 1); i > 0; i -= 1) {
        let randomIndex ;
        if (seed) {
            randomIndex = Math.floor(seed() * (i + 1));
        }
        else {
            randomIndex = Math.floor(Math.random() * (i + 1));
        }
        
        [shuffled[i], shuffled[randomIndex]] = [shuffled[randomIndex],shuffled[i]];
    }
  
    return shuffled;
}

export class TeachablePoseNet extends CustomPoseNet {
    /**
     * Training and validation datasets
     */
    private trainDataset: tf.data.Dataset<TensorContainer>;
    private validationDataset: tf.data.Dataset<TensorContainer>;

    // Number of total samples
    private totalSamples: number = 0;

    // Array of all the examples collected
    public examples: Array<Array<Float32Array>> = [];

    // Optional seed to make shuffling of data predictable
    private seed: seedrandom.prng;

    /**
     * has the teachable model been trained?
     */
    public get isTrained() {
        return !!this.model && this.model.layers && this.model.layers.length > 2;
    }

    /**
     * has the dataset been prepared with all labels and samples processed?
     */
    public get isPrepared() {
        return !!this.trainDataset;
    }
    /**
     * how many classes are in the dataset?
     */
    public get numClasses() {
        return this._metadata.labels.length;
    }

    constructor(public model: tf.LayersModel, public posenetModel: PoseNet, metadata: Partial<Metadata>) {
        super(model, posenetModel, metadata);
    }

    /**
     * Add a sample of data under the provided className
     * @param className the classification this example belongs to
     * @param sample the image / tensor that belongs in this classification
     */
    // public async addExample(className: number, sample: HTMLCanvasElement | tf.Tensor) {
    public async addExample(className: number, sample: Float32Array) {
        // TODO: Do I need to normalize or flip image?
        // const cap = isTensor(sample) ? sample : capture(sample);
        // const example = this.posenet.predict(cap) as tf.Tensor;
        // const embeddingsArray = await this.predictPosenet(sample);

        // save samples of each class separately
        this.examples[className].push(sample);

        // increase our sample counter
        this.totalSamples++;
    }

    /**
     * Classify an input image / Tensor with your trained model
     * @param image the input image / Tensor to classify against your model
     * @param topK how many of the top results do you want? defautls to 3
     */
    public async predict(
        poseOutput: Float32Array, 
        flipped = false, 
        maxPredictions = 3) {
        if (!this.model) {
            throw new Error('Model has not been trained yet, called train() first');
        }

        return super.predict(poseOutput, flipped, maxPredictions);
    }

    /**
     * process the current examples provided to calculate labels and format
     * into proper tf.data.Dataset
     */
    public prepare() {
        for (let classes in this.examples){
            if (classes.length == 0) {
                throw new Error('Add some examples before training');
            }
        }

        const datasets = this.convertToTfDataset();
        this.trainDataset = datasets.trainDataset;
        this.validationDataset = datasets.validationDataset;
    }

    /**
     * Process the examples by first shuffling randomly per class, then adding
     * one-hot labels, then splitting into training/validation datsets, and finally
     * sorting one last time
     */
    private convertToTfDataset() {
        // first shuffle each class individually
        // TODO: we could basically replicate this by insterting randomly
        for (let i = 0; i < this.examples.length; i++) {
            this.examples[i] = fisherYates(this.examples[i], this.seed) as Float32Array[];
        }

        // then break into validation and test datasets

        let trainDataset: Array<Sample> = [];
        let validationDataset: Array<Sample> = [];

        // for each class, add samples to train and validation dataset
        for (let i = 0; i < this.examples.length; i++) {
            const y = flatOneHot(i, this.numClasses);

            const classLength = this.examples[i].length;
            const numValidation = Math.ceil(VALIDATION_FRACTION * classLength);
            const numTrain = classLength - numValidation;

            let classTrain = this.examples[i].slice(0, numTrain).map((dataArray) => {
                return { data: dataArray, label: y };
            });

            let classValidation = this.examples[i].slice(numTrain).map((dataArray) => {
                return { data: dataArray, label: y };
            });

            trainDataset = trainDataset.concat(classTrain);
            validationDataset = validationDataset.concat(classValidation);
        }

        // finally shuffle both train and validation datasets
        trainDataset = fisherYates(trainDataset, this.seed) as Sample[];
        validationDataset = fisherYates(validationDataset, this.seed) as Sample[];

        const trainX = tf.data.array(trainDataset.map(sample => sample.data));
        const validationX = tf.data.array(validationDataset.map(sample => sample.data));
        const trainY = tf.data.array(trainDataset.map(sample => sample.label));
        const validationY = tf.data.array(validationDataset.map(sample => sample.label));
    
        // return tf.data dataset objects
        return { 
            trainDataset: tf.data.zip({ xs: trainX,  ys: trainY}), 
            validationDataset: tf.data.zip({ xs: validationX,  ys: validationY})
        }
    }

    /**
     * Train your data into a new model and join it with mobilenet
     * @param params the parameters for the model / training
     * @param callbacks provide callbacks to receive training events
     */
    public async train(params: TrainingParameters, callbacks: CustomCallbackArgs = {}) {
        if (!this.isPrepared) {
            this.prepare();
        }
        const numLabels = this.getLabels().length;

        util.assert(
            numLabels === this.numClasses,
            () => `Can not train, has ${numLabels} labels and ${this.numClasses} classes`);
        
        // Inputs for posenet
        const inputSize = this.examples[0][1].length;    

        // in case we need to use a seed for predictable training
        let varianceScaling;
        if (this.seed) {
            varianceScaling = tf.initializers.varianceScaling({ seed: 3.14});
        }
        else {
            varianceScaling = tf.initializers.varianceScaling({});
        }

        const trainingModel = tf.sequential({
            layers: [
            // Layer 1.
            tf.layers.dense({
                inputShape: [inputSize],
                units: params.denseUnits,
                activation: 'relu',
                kernelInitializer: varianceScaling, // 'varianceScaling'
                useBias: true
            }),
            // Layer 2. The number of units of the last layer should correspond
            // to the number of classes we want to predict.
            tf.layers.dense({
                units: this.numClasses,
                kernelInitializer: varianceScaling, // 'varianceScaling'
                useBias: false,
                activation: 'softmax'
            })
            ]
        });
        const optimizer = tf.train.adam(params.learningRate);
        trainingModel.compile({ 
            optimizer, 
            loss: 'categoricalCrossentropy', 
            metrics: ['accuracy'] 
        });

        if (!(params.batchSize > 0)) {
            throw new Error(
            `Batch size is 0 or NaN. Please choose a non-zero fraction`
            );
        }
        
        const trainData = this.trainDataset.batch(params.batchSize);
        const validationData = this.validationDataset.batch(params.batchSize);

        // For debugging: check for shuffle or result from trainDataset
        /*
        await trainDataset.forEach((e: tf.Tensor[]) => {
            console.log(e);
            // @ts-ignore
            let data = e.ys.dataSync() as Float32Array;
            console.log(data);
        });
        */
        const history = await trainingModel.fitDataset(trainData, {
            epochs: params.epochs,
            validationData: validationData,
            callbacks
        });

        this.model = trainingModel;
        return this.model;
    }

    /*
     * Setup the exampls array to hold samples per class
     */
    public prepareDataset() {
        for (let i = 0; i < this.numClasses; i++) {
            this.examples[i] = [];
        }
    }

    public setLabel(index: number, label: string) {
        this._metadata.labels[index] = label;
    }

    public setLabels(labels: string[]) {
        this._metadata.labels = labels;
        this.prepareDataset();
    }

    public getLabel(index: number) {
        return this._metadata.labels[index];
    }

    public getLabels() {
        return this._metadata.labels;
    }

    public setName(name: string) {
        this._metadata.modelName = name;
    }

    public getName() {
        return this._metadata.modelName;
    }

    /* 
     * optional seed for predictable shuffling of dataset
     */
    public setSeed(seed: string) {
        this.seed = seedrandom(seed);
    }
}

export async function createTeachable(metadata: Partial<Metadata>) {
    const posenetModel = await loadPoseNet();

    return new TeachablePoseNet(tf.sequential(), posenetModel, metadata);
}