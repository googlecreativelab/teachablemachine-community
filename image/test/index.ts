import { assert } from 'chai';

import * as tf from '@tensorflow/tfjs';
import * as tm from '../src/index';

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


// benchmark('test', =>{

// })

describe('Train a custom model', () => {
    it('create a model', async ()=>{
        const teachableMobileNet = await tm.mobilenet.createTeachable({
            tfjsVersion: tf.version.tfjs,
            // tmVersion: version
        });
        assert.exists(teachableMobileNet);
    });

    it('Train flower dataset', async ()=>{
        const DATASET_TRAIN_SIZE = 30;
        const DATASET_VALIDATION_SIZE = 30;
        const teachableMobileNet = await tm.mobilenet.createTeachable({
            tfjsVersion: tf.version.tfjs,
        });

        const classes = ['daisy','dandelion','roses','sunflowers','tulips'];
        teachableMobileNet.setLabels(classes);

        const images:HTMLImageElement[][] = [];
        for(const c of classes){
            const load: Array<Promise<HTMLImageElement>> = [];
            for(let i=0;i<DATASET_TRAIN_SIZE;i++){
                load.push(loadFlowerImage(c, i));
            }
            images.push(
                await Promise.all(load)
            );
        }

        

        
        await tf.nextFrame().then(async () => {
            let index = 0;
            for(const imgSet of images){
                for(const img of imgSet){
                    teachableMobileNet.addExample(index, img);                
                }
                index ++;
            }
            console.log("data loaded");
            console.time("Train");
            await teachableMobileNet.train({
                denseUnits: 100,
                epochs:20,
                learningRate: 0.0001
            },
            {
                onBatchEnd: async (batch, logs) => {
                    console.log(logs);
                },
                onEpochBegin: async (epoch, logs) => {
                    console.log('Epoch: ', epoch);
                },
                onEpochEnd: async (epoch, logs) => {
                    console.log(logs);
                    console.log('\n');
                }
            });
            console.timeEnd("Train");            
        });


        // Validation
        let accuracy = 0;
        for(const c of classes){
            let s = 0;
            for(let i=DATASET_TRAIN_SIZE; i<DATASET_TRAIN_SIZE+DATASET_VALIDATION_SIZE; i++){
                const testImage = await loadFlowerImage(c,i);
                const scores = await teachableMobileNet.predict(testImage, false);
                if(scores[0].className === c){
                    s ++;
                }
            }
            console.log(c, s/DATASET_VALIDATION_SIZE);
            accuracy += s/DATASET_VALIDATION_SIZE;
        }
        assert.isTrue(accuracy > 0.6);
        console.log("Final accuracy", accuracy/classes.length);
    
    
    }).timeout(240000);
    
});