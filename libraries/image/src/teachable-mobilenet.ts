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
import { util, Rank } from '@tensorflow/tfjs';
import { capture } from './utils/tf';
import { TensorContainer } from '@tensorflow/tfjs-core/dist/tensor_types';
import { CustomCallbackArgs } from '@tensorflow/tfjs';
import { CustomMobileNet,
    Metadata,
    loadTruncatedMobileNet,
    ClassifierInputSource,
    ModelOptions
} from './custom-mobilenet';
import * as seedrandom from 'seedrandom';
import { Initializer } from '@tensorflow/tfjs-layers/dist/initializers';

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
    typeof c.dataId === 'object' && typeof c.shape === 'object';

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
    const shuffled = array.slice();

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

export class TeachableMobileNet extends CustomMobileNet {
    /**
     * the training model for transfer learning
     */
    protected trainingModel: tf.LayersModel;

    /**
     * Training and validation datasets
     */
    private trainDataset: tf.data.Dataset<TensorContainer>;
    private validationDataset: tf.data.Dataset<TensorContainer>;

    private __stopTrainingResolve: () => void;
    // private __stopTrainingReject: (error: Error) => void;

    // Number of total samples
    private totalSamples = 0;

    // Array of all the examples collected
    public examples: Float32Array[][] = [];

    // Optional seed to make shuffling of data predictable
    private seed: seedrandom.prng;

    constructor(truncated: tf.LayersModel, metadata: Partial<Metadata>) {
        super(tf.sequential(), metadata);
        // the provided model is the truncated mobilenet
        this.truncatedModel = truncated;
    }

    public get asSequentialModel() {
        return this.model as tf.Sequential;
    }


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
    public get numClasses(): number {
        return this._metadata.labels.length;
    }

    /**
     * Add a sample of data under the provided className
     * @param className the classification this example belongs to
     * @param sample the image / tensor that belongs in this classification
     */
    // public async addExample(className: number, sample: HTMLCanvasElement | tf.Tensor) {
    public async addExample(className: number, sample: HTMLImageElement | HTMLCanvasElement | tf.Tensor) {
        const cap = isTensor(sample) ? sample : capture(sample, this._metadata.grayscale);
        const example = this.truncatedModel.predict(cap) as tf.Tensor;

        const activation = example.dataSync() as Float32Array;
        cap.dispose();
        example.dispose();

        // save samples of each class separately
        this.examples[className].push(activation);

        // increase our sample counter
        this.totalSamples++;
    }

    /**
     * Classify an input image / Tensor with your trained model. Return all results.
     * @param image the input image / Tensor to classify against your model
     * @param topK how many of the top results do you want? defautls to 3
     */
    public async predict(image: ClassifierInputSource, flipped = false) {
        if (!this.model) {
            throw new Error('Model has not been trained yet, called train() first');
        }
        return super.predict(image, flipped);
    }

    /**
     * Classify an input image / Tensor with your trained model. Return topK results
     * @param image the input image / Tensor to classify against your model
     * @param maxPredictions how many of the top results do you want? defautls to 3
     * @param flipped whether to flip an image
     */
    public async predictTopK(image: ClassifierInputSource, maxPredictions = 10, flipped = false, ) {
        if (!this.model) {
            throw new Error('Model has not been trained yet, called train() first');
        }
        return super.predictTopK(image, maxPredictions, flipped);
    }

    /**
     * process the current examples provided to calculate labels and format
     * into proper tf.data.Dataset
     */
    public prepare() {
        for (const classes in this.examples){
            if (classes.length === 0) {
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

        let trainDataset: Sample[] = [];
        let validationDataset: Sample[] = [];

        // for each class, add samples to train and validation dataset
        for (let i = 0; i < this.examples.length; i++) {
            const y = flatOneHot(i, this.numClasses);

            const classLength = this.examples[i].length;
            const numValidation = Math.ceil(VALIDATION_FRACTION * classLength);
            const numTrain = classLength - numValidation;

            const classTrain = this.examples[i].slice(0, numTrain).map((dataArray) => {
                return { data: dataArray, label: y };
            });

            const classValidation = this.examples[i].slice(numTrain).map((dataArray) => {
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
        };
    }

    /**
     * Saving `model`'s topology and weights as two files
     * (`my-model-1.json` and `my-model-1.weights.bin`) as well as
     * a `metadata.json` file containing metadata such as text labels to be
     * downloaded from browser.
     * @param handlerOrURL An instance of `IOHandler` or a URL-like,
     * scheme-based string shortcut for `IOHandler`.
     * @param config Options for saving the model.
     * @returns A `Promise` of `SaveResult`, which summarizes the result of
     * the saving, such as byte sizes of the saved artifacts for the model's
     *   topology and weight values.
     */
    public async save(handlerOrURL: tf.io.IOHandler | string, config?: tf.io.SaveConfig): Promise<tf.io.SaveResult> {
        return this.model.save(handlerOrURL, config);
    }

    /**
     * Train your data into a new model and join it with mobilenet
     * @param params the parameters for the model / training
     * @param callbacks provide callbacks to receive training events
     */
    public async train(params: TrainingParameters, callbacks: CustomCallbackArgs = {}) {
        // Add callback for onTrainEnd in case of early stop
        const originalOnTrainEnd = callbacks.onTrainEnd || (() => {});
        callbacks.onTrainEnd = (logs: tf.Logs) => {
            if (this.__stopTrainingResolve) {
                this.__stopTrainingResolve();
                this.__stopTrainingResolve = null;
            }
            originalOnTrainEnd(logs);
        };
        
        // Rest of trian function
        if (!this.isPrepared) {
            this.prepare();
        }

        const numLabels = this.getLabels().length;
        util.assert(
            numLabels === this.numClasses,
            () => `Can not train, has ${numLabels} labels and ${this.numClasses} classes`);

        const inputShape = this.truncatedModel.outputs[0].shape.slice(1); // [ 7 x 7 x 1280]
        const inputSize = tf.util.sizeFromShape(inputShape);

        // in case we need to use a seed for predictable training
        let varianceScaling: Initializer;
        if (this.seed) {
            varianceScaling = tf.initializers.varianceScaling({ seed: 3.14});
        }
        else {
            varianceScaling = tf.initializers.varianceScaling({});
        }

        this.trainingModel = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [inputSize],
                    units: params.denseUnits,
                    activation: 'relu',
                    kernelInitializer: varianceScaling, // 'varianceScaling'
                    useBias: true
                }),
                tf.layers.dense({
                    kernelInitializer: varianceScaling, // 'varianceScaling'
                    useBias: false,
                    activation: 'softmax',
                    units: this.numClasses
                })
            ]
        });

        const optimizer = tf.train.adam(params.learningRate);
        // const optimizer = tf.train.rmsprop(params.learningRate);

        this.trainingModel.compile({
            optimizer,
            // loss: 'binaryCrossentropy',
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
        })
        */

        const history = await this.trainingModel.fitDataset(trainData, {
            epochs: params.epochs,
            validationData,
            callbacks
        });

        const jointModel = tf.sequential();
        jointModel.add(this.truncatedModel);
        jointModel.add(this.trainingModel);
        this.model = jointModel;

        optimizer.dispose(); // cleanup of memory

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

    public stopTraining() {  
        const promise = new Promise((resolve, reject) => {
            this.trainingModel.stopTraining = true;
            this.__stopTrainingResolve = resolve;
            // this.__stopTrainingReject = reject;
        });
        
        return promise;
    }

    public dispose() {
        this.trainingModel.dispose();
        super.dispose();
    }

    /* 
     * Calculate each class accuracy using the validation dataset
     */
    public async calculateAccuracyPerClass() {
        const validationXs = this.validationDataset.mapAsync(async (dataset: TensorContainer) => {
            return (dataset as { xs: TensorContainer, ys: TensorContainer}).xs;
        });
        const validationYs = this.validationDataset.mapAsync(async (dataset: TensorContainer) => {
            return (dataset as { xs: TensorContainer, ys: TensorContainer}).ys;
        });

        // we need to split our validation data into batches in case it is too large to fit in memory
        const batchSize = Math.min(validationYs.size, 32);
        const iterations = Math.ceil(validationYs.size / batchSize);

        const batchesX = validationXs.batch(batchSize);
        const batchesY = validationYs.batch(batchSize);
        const itX = await batchesX.iterator();
        const itY = await batchesY.iterator();
        const allX = [];
        const allY = [];

        for (let i = 0; i < iterations; i++) {
            // 1. get the prediction values in batches
            const batchedXTensor = await itX.next();
            const batchedXPredictionTensor = this.trainingModel.predict(batchedXTensor.value) as tf.Tensor;
            const argMaxX = batchedXPredictionTensor.argMax(1); // Returns the indices of the max values along an axis
            allX.push(argMaxX);

            // 2. get the ground truth label values in batches
            const batchedYTensor = await itY.next();
            const argMaxY = batchedYTensor.value.argMax(1); // Returns the indices of the max values along an axis
            allY.push(argMaxY);

            // 3. dispose of all our tensors
            batchedXTensor.value.dispose();
            batchedXPredictionTensor.dispose();
            batchedYTensor.value.dispose();
        }

        // concatenate all the results of the batches
        const reference = tf.concat(allY); // this is the ground truth
        const predictions = tf.concat(allX); // this is the prediction our model is guessing

        // only if we concatenated more than one tensor for preference and reference
        if (iterations !== 1) {
            for (let i = 0; i < allX.length; i++) {
                allX[i].dispose();
                allY[i].dispose();
            }
        }

        return { reference, predictions };
    }

    /*
     * optional seed for predictable shuffling of dataset
     */
    public setSeed(seed: string) {
        this.seed = seedrandom(seed);
    }
}

export async function createTeachable(metadata: Partial<Metadata>, modelOptions?: ModelOptions) {
    const mobilenet = await loadTruncatedMobileNet(modelOptions);
    return new TeachableMobileNet(mobilenet, metadata);
}
