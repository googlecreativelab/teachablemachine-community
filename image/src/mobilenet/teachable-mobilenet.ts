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
import { util } from '@tensorflow/tfjs';
import { capture } from '../utils/tf';
import { TensorContainer } from '@tensorflow/tfjs-core/dist/tensor_types';
import { CustomCallbackArgs, equalStrict } from '@tensorflow/tfjs';
import { CustomMobileNet, Metadata, loadTruncatedMobileNet, ClassifierInputSource } from './custom-mobilenet';


export interface TrainingParameters {
    denseUnits: number;
    epochs: number;
    learningRate: number;
}

// tslint:disable-next-line:no-any
const isTensor = (c: any): c is tf.Tensor =>
    typeof c.dataId === 'object' && c.shape === 'object';


/**
 * Converts an integer into its one-hot representation and returns
 * the data as a JS Array.
 */
function flatOneHot(label: number, numClasses: number) {
    const labelOneHot = new Array(numClasses).fill(0);
    labelOneHot[label] = 1;

    return labelOneHot;
}

function convertToTfDataset(xs: Float32Array[], ys: number[][]) {
    const xTrain = tf.data.array(xs);
    const yTrain = tf.data.array(ys);

    const trainDataset = tf.data.zip({ xs: xTrain,  ys: yTrain});

    //TODO: ys.length might not always be best
    const shuffled = trainDataset.shuffle(ys.length);

    return shuffled;
}

export class TeachableMobileNet extends CustomMobileNet {

    /**
     * the truncated mobilenet model we will train on top of
     */
    protected truncatedModel: tf.LayersModel;

    public get asSequentialModel() {
        return this.model as tf.Sequential;
    }


    // Array<[className, activation]>
    public examples: Array<[number, Float32Array]> = [];
    private ys: number[][];
    private dataset: tf.data.Dataset<TensorContainer>;

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
        return !!this.dataset;
    }

    /**
     * how many classes are in the dataset?
     */
    public get numClasses() {
        // get the highest provided className
        return Math.max(...this.examples.map(ex => ex[0])) + 1;
    }

    constructor(truncated: tf.LayersModel, metadata: Partial<Metadata>) {
        super(tf.sequential(), metadata);
        // the provided model is the truncated mobilenet
        this.truncatedModel = truncated;
    }

    /**
     * Add a sample of data under the provided className
     * @param className the classification this example belongs to
     * @param sample the image / tensor that belongs in this classification
     */
    // public async addExample(className: number, sample: HTMLCanvasElement | tf.Tensor) {
    public async addExample(className: number, sample: HTMLCanvasElement | tf.Tensor) {
        const cap = isTensor(sample) ? sample : capture(sample);
        const example = this.truncatedModel.predict(cap) as tf.Tensor;

        const activation = example.dataSync() as Float32Array;

        cap.dispose();
        this.examples.push([ className, activation ]);

        // we dont have a dataset if we just changed the data examples
        this.dataset = null;
    }

    /**
     * Classify an input image / Tensor with your trained model
     * @param image the input image / Tensor to classify against your model
     * @param topK how many of the top results do you want? defautls to 3
     */
    public async predict(image: ClassifierInputSource, flipped = false, maxPredictions = 3) {
        if (!this.model) {
            throw new Error('Model has not been trained yet, called train() first');
        }
        return super.predict(image, flipped, maxPredictions);
    }


    /**
     * process the current examples provided to calculate labels and format
     * into proper tf.data.Dataset
     */
    public prepare() {
        const xs: Float32Array[] = this.examples.map(ex => ex[1]);
        if (!xs.length) {
            throw new Error('Add some examples before training');
        }
        const ys: number[][] = [];
        const numClasses = this.numClasses;
        for ( const [label] of this.examples) {
            ys.push(flatOneHot(label, numClasses));
        }

        this.dataset = convertToTfDataset(xs, ys);
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

        // Approach 1 in dataset.ts
        const inputShape = this.truncatedModel.outputs[0].shape.slice(1); // [ 7 x 7 x 256]
        const inputSize = tf.util.sizeFromShape(inputShape);

        // Creates a 2-layer fully connected model. By creating a separate model,
        // rather than adding layers to the mobilenet model, we "freeze" the weights
        // of the mobilenet model, and only train weights from the new model.
        const trainingModel = tf.sequential({
            layers: [
            // Layer 1.
            tf.layers.dense({
                inputShape: [inputSize],
                units: params.denseUnits,
                activation: 'relu',
                kernelInitializer: 'varianceScaling',
                useBias: true
            }),
            // Layer 2. The number of units of the last layer should correspond
            // to the number of classes we want to predict.
            tf.layers.dense({
                units: this.numClasses,
                kernelInitializer: 'varianceScaling',
                useBias: false,
                activation: 'softmax'
            })
            ]
        });

        const optimizer = tf.train.adam(params.learningRate);
        trainingModel.compile({ optimizer, loss: 'categoricalCrossentropy' });

        //const batchSize = Math.floor(dataset.xs.shape[0] * trainParams.getBatchSizeFraction())
        const batchSize = Math.min(16, this.examples.length);

        if (!(batchSize > 0)) {
            throw new Error(
            `Batch size is 0 or NaN. Please choose a non-zero fraction`
            );
        }

        const trainDataset = this.dataset.batch(batchSize);

        // For debugging: check for shuffle or result from trainDataset
        /*
        await trainDataset.forEach((e: tf.Tensor[]) => {
            console.log(e);
        })
        */

        const history = await trainingModel.fitDataset(trainDataset, {
            epochs: params.epochs,
            callbacks
        });

        const jointModel = tf.sequential();
        jointModel.add(this.truncatedModel);
        jointModel.add(tf.layers.flatten());
        jointModel.add(trainingModel);

        this.model = jointModel;

        return this.model;
    }

    public setLabel(index: number, label: string) {
        this._metadata.labels[index] = label;
    }

    public setLabels(labels: string[]) {
        this._metadata.labels = labels;
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
}

export async function createTeachable(metadata: Partial<Metadata>, checkpoint?: string) {
    const mobilenet = await loadTruncatedMobileNet(checkpoint);
    return new TeachableMobileNet(mobilenet, metadata);
}
