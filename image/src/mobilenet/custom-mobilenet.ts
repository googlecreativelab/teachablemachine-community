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
import { capture } from '../utils/tf';
import { cropTo } from '../utils/canvas';
import { version } from '../version';
import { conv2dTranspose } from '@tensorflow/tfjs-layers/dist/exports_layers';


const MOBILENET_MODEL_PATH =
    // tslint:disable-next-line:max-line-length
    'https://storage.googleapis.com/teachable-machine-models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224_no_top/model.json';
    // tslint:disable-next-line:max-line-length
    // 'https://storage.googleapis.com/teachable-machine-models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224/model.json';
    // tslint:disable-next-line:max-line-length
    // 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

const TRAINING_LAYER = 'out_relu'; //MobileNetV2
// const TRAINING_LAYER = 'conv_pw_13_relu'; // MobilenetV1

export const IMAGE_SIZE = 224;

/**
 * the metadata to describe the model's creation,
 * includes the labels associated with the classes
 * and versioning information from training.
 */
export interface Metadata {
    tfjsVersion: string;
    tmVersion?: string;
    tmSupportVersion: string;
    modelName?: string;
    timeStamp?: string;
    labels: string[];
    userMetadata?: {};
}

/**
 * Receives a Metadata object and fills in the optional fields such as timeStamp
 * @param data a Metadata object
 */
const fillMetadata = (data: Partial<Metadata>) => {
    util.assert(typeof data.tfjsVersion === 'string', () => `metadata.tfjsVersion is invalid`);
    data.tmSupportVersion = data.tmSupportVersion || version;
    data.timeStamp = data.timeStamp || new Date().toISOString();
    data.userMetadata = data.userMetadata || {};
    data.modelName = data.modelName || 'untitled';
    data.labels = data.labels || [];
    return data as Metadata;
};

// tslint:disable-next-line:no-any
const isMetadata = (c: any): c is Metadata =>
    !!c && typeof c.tmVersion === 'string' &&
    typeof c.tmSupportVersion === 'string' && Array.isArray(c.labels);

// export const toMetadata = (
//    tfjsVersion: string,
//    tmVersion: string, labels: string[], name = 'tm-pro') => {
        // return {
        //     tfjsVersion,
        //     tmVersion,
        //     tmSupportVersion: version,
        //     modelName: name,
        //     timeStamp: new Date().toISOString(),
        //     labels: labels
        // };
// };

/**
 * process either a URL string or a Metadata object
 * @param metadata a url to load metadata or a Metadata object
 */
const processMetadata = async (metadata: string | Metadata) => {
    let metadataJSON: Metadata;
    if (typeof metadata === 'string') {
        util.assert(
            metadata.indexOf('http') === 0,
            () => 'metadata is a string but not a valid url'
        );
        metadataJSON = await (await fetch(metadata)).json();
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
     * Given an image element, makes a prediction through mobilenet returning the
     * probabilities of the top K classes.
     * @param image the image to classify
     * @param maxPredictions the maximum number of classification predictions
     */
    async predict( image: ClassifierInputSource, flipped = false, maxPredictions = 10) {
        const croppedImage = cropTo(image, IMAGE_SIZE, flipped);

        const logits = tf.tidy(() => {
            const captured = capture(croppedImage);
            return this.model.predict(captured);
        });

        // Convert logits to probabilities and class names.
        const classes = await getTopKClasses(this._metadata.labels, logits as tf.Tensor<tf.Rank>, maxPredictions);
        dispose(logits);

        return classes;
    }
}


/**
 * load the base mobilenet model
 * @param checkpoint the URL to load the mobilenet json
 */
export async function loadTruncatedMobileNet(checkpoint: string = MOBILENET_MODEL_PATH) {
    const mobilenet = await tf.loadLayersModel(checkpoint);
    const layer = mobilenet.getLayer(TRAINING_LAYER);
    const truncatedModel = tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
    return truncatedModel;
}

export async function load(checkpoint: string, metadata?: string | Metadata ) {
    const customModel = await tf.loadLayersModel(checkpoint);
    const metadataJSON = metadata ? await processMetadata(metadata) : null;
    return new CustomMobileNet(customModel, metadataJSON);
}

export async function loadFromFiles(json: File, weights: File, metadata?: string | Metadata) {
    const customModel = await tf.loadLayersModel(tf.io.browserFiles([json, weights]));
    const metadataJSON = metadata ? await processMetadata(metadata) : null;
    return new CustomMobileNet(customModel, metadataJSON);
}
