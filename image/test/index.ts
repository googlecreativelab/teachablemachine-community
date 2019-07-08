import { assert } from 'chai';

import * as tf from '@tensorflow/tfjs';
import * as tm from '../src/index';
import { TeachableMobileNet } from '../src/mobilenet';

function loadFlowerImage(c:string, i:number):Promise<HTMLImageElement>{
    // tslint:disable-next-line:max-line-length
    const src = `https://storage.googleapis.com/teachable-machine-models/test_data/flowers_200/class-${c}-image-model/${i}.png`;
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.crossOrigin = "anonymous";
      img.src = src;
    });
}

async function testModel(
  model: TeachableMobileNet,
  trainSize: number,
  validationSize: number,
  loadFunction: Function,
  classes: string[],
  epochs = 20
) {
  model.setLabels(classes);

  const images: HTMLImageElement[][] = [];
  for (const c of classes) {
    const load: Array<Promise<HTMLImageElement>> = [];
    for (let i = 0; i < trainSize; i++) {
      load.push(loadFunction(c, i));
    }
    images.push(await Promise.all(load));
  }

  await tf.nextFrame().then(async () => {
    let index = 0;
    for (const imgSet of images) {
      for (const img of imgSet) {
        model.addExample(index, img);
      }
      index++;
    }
    console.log("data loaded");
    console.time("Train");
    await model.train(
      {
        denseUnits: 100,
        epochs,
        learningRate: 0.0001
      },
      {
        onBatchEnd: async (batch, logs) => {
        },
        onEpochBegin: async (epoch, logs) => {
          console.log("Epoch: ", epoch);
        },
        onEpochEnd: async (epoch, logs) => {
          console.log(logs);
        }
      }
    );
    console.timeEnd("Train");
  });

  // Validation
  let accuracy = 0;
  for (const c of classes) {
    let s = 0;
    for (let i = trainSize; i < trainSize + validationSize; i++) {
      const testImage = await loadFlowerImage(c, i);
      const scores = await model.predict(testImage, false);
      if (scores[0].className === c) {
        s++;
      }
    }
    console.log(c, s / validationSize);
    accuracy += s / validationSize;
  }
  return accuracy;
}

// Weird workaround...
tf.util.fetch = (a,b)=> window.fetch(a,b);

describe('Beginning', () => {
    it('happens', () => {
        assert.equal(typeof tm, 'object', 'tm should be an object');
        assert.equal(typeof tm.getWebcam, 'function', 'tm.getWebcam should be a function');
        assert.equal(typeof tm.version, 'string', 'tm.version should be a string');
        assert.equal(tm.mobilenet.IMAGE_SIZE, 224, 'IMAGE_SIZE should be 224');
    });
});


describe('Train a custom model', () => {
    it('create a model', async ()=>{
        const teachableMobileNet = await tm.mobilenet.createTeachable({
            tfjsVersion: tf.version.tfjs,
            // tmVersion: version
        });
        assert.exists(teachableMobileNet);
    });

    it('Train flower dataset on mobilenet v2', async ()=>{
        const DATASET_TRAIN_SIZE = 30;
        const DATASET_VALIDATION_SIZE = 30;
        const CLASSES = ['daisy','dandelion','roses','sunflowers','tulips'];
        
        const teachableMobileNetV2 = await tm.mobilenet.createTeachable({
            tfjsVersion: tf.version.tfjs,            
        }, { 
            version: 2 
        });
          
        const accuracyV2 = await testModel(
          teachableMobileNetV2,
          DATASET_TRAIN_SIZE,
          DATASET_VALIDATION_SIZE,
          loadFlowerImage,
          CLASSES
        );       
        assert.isTrue(accuracyV2 > 0.6);
        console.log("Final accuracy MobileNetV2", accuracyV2/CLASSES.length);
    
    }).timeout(240000);

    it('Train flower dataset on mobilenet v1', async ()=>{
        const DATASET_TRAIN_SIZE = 30;
        const DATASET_VALIDATION_SIZE = 30;
        const CLASSES = ['daisy','dandelion','roses','sunflowers','tulips'];
        
        const teachableMobileNetV1 = await tm.mobilenet.createTeachable({
            tfjsVersion: tf.version.tfjs,            
        }, { 
            version: 1 
        });

        const accuracyV1 = await testModel(
            teachableMobileNetV1,
            DATASET_TRAIN_SIZE,
            DATASET_VALIDATION_SIZE,
            loadFlowerImage,
            CLASSES
          );       
          assert.isTrue(accuracyV1 > 0.6);
          console.log("Final accuracy MobileNetV1", accuracyV1/CLASSES.length);
    }).timeout(240000);
    
});