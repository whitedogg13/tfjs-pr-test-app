import * as tf from '@tensorflow/tfjs';

const MOBILENET_MODEL_PATH =
  // tslint:disable-next-line:max-line-length
  "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1";

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 10;

async function run() {
  console.log('Loading model...');

  mobilenet = await tf.loadGraphModel(MOBILENET_MODEL_PATH, {fromTFHub: true});

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  console.log('');
}

run();
