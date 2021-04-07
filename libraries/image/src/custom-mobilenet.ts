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
import { util, SymbolicTensor } from '@tensorflow/tfjs';
import { dispose } from '@tensorflow/tfjs';
import { capture } from './utils/tf';
import { cropTo } from './utils/canvas';
import { version } from './version';

const DEFAULT_MOBILENET_VERSION = 1;
const DEFAULT_TRAINING_LAYER_V1 = 'conv_pw_13_relu';
const DEFAULT_TRAINING_LAYER_V2 = "out_relu"; 
const DEFAULT_ALPHA_V1 = 0.25;
const DEFAULT_ALPHA_V2 = 0.35;
export const IMAGE_SIZE = 224;

/**
 * the metadata to describe the model's creation,
 * includes the labels associated with the classes
 * and versioning information from training.
 */
export interface Metadata {
    tfjsVersion: string;
    tmVersion?: string;
    packageVersion: string;
    packageName: string;
    modelName?: string;
    timeStamp?: string;
    labels: string[];
    userMetadata?: {};
    grayscale?: boolean;
    imageSize?: number;
}

export interface ModelOptions {
    version?: number;
    checkpointUrl?: string;
    alpha?: number;
    trainingLayer?: string;
}

/**
 * Receives a Metadata object and fills in the optional fields such as timeStamp
 * @param data a Metadata object
 */
const fillMetadata = (data: Partial<Metadata>) => {
    // util.assert(typeof data.tfjsVersion === 'string', () => `metadata.tfjsVersion is invalid`);
    data.packageVersion = data.packageVersion || version;
    data.packageName = data.packageName || '@teachablemachine/image';
    data.timeStamp = data.timeStamp || new Date().toISOString();
    data.userMetadata = data.userMetadata || {};
    data.modelName = data.modelName || 'untitled';
    data.labels = data.labels || [];
    data.imageSize = data.imageSize || IMAGE_SIZE;
    return data as Metadata;
};

// tslint:disable-next-line:no-any
const isMetadata = (c: any): c is Metadata =>
    !!c && Array.isArray(c.labels);

const isAlphaValid = (version: number, alpha: number) => {
    if (version === 1) {
        if (alpha !== 0.25 && alpha !== 0.5 && alpha !== 0.75 && alpha !== 1) {
            console.warn("Invalid alpha. Options are: 0.25, 0.50, 0.75 or 1.00.");
            console.log("Loading model with alpha: ", DEFAULT_ALPHA_V1.toFixed(2)); 
            return DEFAULT_ALPHA_V1;
        }
    }
    else {
        if (alpha !== 0.35 && alpha !== 0.5 && alpha !== 0.75 && alpha !== 1) {
            console.warn("Invalid alpha. Options are: 0.35, 0.50, 0.75 or 1.00.");
            console.log("Loading model with alpha: ", DEFAULT_ALPHA_V2.toFixed(2)); 
            return DEFAULT_ALPHA_V2;
        }
    }

    return alpha;
};

const parseModelOptions = (options?: ModelOptions) => {
    options = options || {}

    if (options.checkpointUrl && options.trainingLayer) {
        if (options.alpha || options.version){
            console.warn("Checkpoint URL passed to modelOptions, alpha options are ignored");
        }        
        return [options.checkpointUrl, options.trainingLayer];
    } else {
        options.version = options.version || DEFAULT_MOBILENET_VERSION;
        
        if(options.version === 1){
            options.alpha = options.alpha || DEFAULT_ALPHA_V1;  
            options.alpha = isAlphaValid(options.version, options.alpha);

            console.log(`Loading mobilenet ${options.version} and alpha ${options.alpha}`);
            // exception is alpha of 1 can only be 1.0
            let alphaString = options.alpha.toFixed(2);
            if (alphaString === "1.00") { alphaString = "1.0"; }

            return [
                // tslint:disable-next-line:max-line-length        
                `https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_${alphaString}_${IMAGE_SIZE}/model.json`,
                DEFAULT_TRAINING_LAYER_V1
            ];
        }
        else if (options.version === 2){
            options.alpha = options.alpha || DEFAULT_ALPHA_V2;  
            options.alpha = isAlphaValid(options.version, options.alpha);

            console.log(`Loading mobilenet ${options.version} and alpha ${options.alpha}`);
            return [
                // tslint:disable-next-line:max-line-length        
                `https://storage.googleapis.com/teachable-machine-models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_${options.alpha.toFixed(2)}_${IMAGE_SIZE}_no_top/model.json`,
                DEFAULT_TRAINING_LAYER_V2
            ];
        } else {
            throw new Error(`MobileNet V${options.version} doesn't exist`);
        }   
    }
};

/**
 * process either a URL string or a Metadata object
 * @param metadata a url to load metadata or a Metadata object
 */
const processMetadata = async (metadata: string | Metadata) => {
    let metadataJSON: Metadata;
    if (typeof metadata === 'string') {
        const metadataResponse = await fetch(metadata);
        metadataJSON = await metadataResponse.json();
    } else if (isMetadata(metadata)) {
        metadataJSON = metadata;
    } else {
        throw new Error('Invalid Metadata provided');
    }
    return fillMetadata(metadataJSON);
};

export type ClassifierInputSource = HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | ImageBitmap;


/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
export async function getTopKClasses(labels: string[], logits: tf.Tensor<tf.Rank>, topK = 3) {
  const values = await logits.data();
  return tf.tidy(() => {
      topK = Math.min(topK, values.length);

      const valuesAndIndices = [];
      for (let i = 0; i < values.length; i++) {
          valuesAndIndices.push({value: values[i], index: i});
      }
      valuesAndIndices.sort((a, b) => {
          return b.value - a.value;
      });
      const topkValues = new Float32Array(topK);
      const topkIndices = new Int32Array(topK);
      for (let i = 0; i < topK; i++) {
          topkValues[i] = valuesAndIndices[i].value;
          topkIndices[i] = valuesAndIndices[i].index;
      }

      const topClassesAndProbs = [];
      for (let i = 0; i < topkIndices.length; i++) {
          topClassesAndProbs.push({
              className: labels[topkIndices[i]], //IMAGENET_CLASSES[topkIndices[i]],
              probability: topkValues[i]
          });
      }
      return topClassesAndProbs;
  });
}


export class CustomMobileNet {
    /**
     * the truncated mobilenet model we will train on top of
     */
    protected truncatedModel: tf.LayersModel;

    static get EXPECTED_IMAGE_SIZE() {
        return IMAGE_SIZE;
    }

    protected _metadata: Metadata;
    public getMetadata() {
        return this._metadata;
    }

    constructor(public model: tf.LayersModel, metadata: Partial<Metadata>) {
        this._metadata = fillMetadata(metadata);
    }

    /**
     * get the total number of classes existing within model
     */
    getTotalClasses() {
        const output = this.model.output as SymbolicTensor;
        const totalClasses = output.shape[1];
        return totalClasses;
    }

    /**
     * get the model labels
     */
    getClassLabels() {
        return this._metadata.labels;
    }

    /**
     * Given an image element, makes a prediction through mobilenet returning the
     * probabilities of the top K classes.
     * @param image the image to classify
     * @param maxPredictions the maximum number of classification predictions
     */
    async predictTopK(image: ClassifierInputSource, maxPredictions = 10, flipped = false) {
        const croppedImage = cropTo(image, this._metadata.imageSize, flipped);

        const logits = tf.tidy(() => {
            const captured = capture(croppedImage, this._metadata.grayscale);
            return this.model.predict(captured);
        });

        // Convert logits to probabilities and class names.
        const classes = await getTopKClasses(this._metadata.labels, logits as tf.Tensor<tf.Rank>, maxPredictions);
        dispose(logits);

        return classes;
    }

    /**
     * Given an image element, makes a prediction through mobilenet returning the
     * probabilities for ALL classes.
     * @param image the image to classify
     * @param flipped whether to flip the image on X
     */
    async predict(image: ClassifierInputSource, flipped = false) {
        const croppedImage = cropTo(image, this._metadata.imageSize, flipped);

        const logits = tf.tidy(() => {
            const captured = capture(croppedImage, this._metadata.grayscale);
            return this.model.predict(captured);
        });

        const values = await (logits as tf.Tensor<tf.Rank>).data();

        const classes = [];
        for (let i = 0; i < values.length; i++) {
            classes.push({
                className: this._metadata.labels[i],
                probability: values[i]
            });
        }

        dispose(logits);

        return classes;
    }

    public dispose() {
        this.truncatedModel.dispose();
    }
}

/**
 * load the base mobilenet model
 * @param modelOptions options determining what model to load
 */
export async function loadTruncatedMobileNet(modelOptions?: ModelOptions) {
    const [checkpointUrl, trainingLayer] = parseModelOptions(modelOptions);
    const mobilenet = await tf.loadLayersModel(checkpointUrl);

    if (modelOptions && modelOptions.version === 1){
        const layer = mobilenet.getLayer(trainingLayer);
        const truncatedModel = tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
        const model = tf.sequential();
        model.add(truncatedModel);
        model.add(tf.layers.flatten());
        return model;
    }
    else {
        const layer = mobilenet.getLayer(trainingLayer);
        const truncatedModel = tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
        const model = tf.sequential();
        model.add(truncatedModel);
        model.add(tf.layers.globalAveragePooling2d({})); // go from shape [7, 7, 1280] to [1280]
        return model;
    }
}

export async function load(model: string, metadata?: string | Metadata ) {
    const customModel = await tf.loadLayersModel(model);
    const metadataJSON = metadata ? await processMetadata(metadata) : null;
    return new CustomMobileNet(customModel, metadataJSON);
}

export async function loadFromFiles(model: File, weights: File, metadata: File) {
    const customModel = await tf.loadLayersModel(tf.io.browserFiles([model, weights]));
    const metadataFile = await new Response(metadata).json();
    const metadataJSON = metadata ? await processMetadata(metadataFile) : null;
    return new CustomMobileNet(customModel, metadataJSON);
}
