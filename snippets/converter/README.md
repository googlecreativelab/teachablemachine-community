# Model Converter

Three Docker images that run on Appengine Flex to convert each type of model. They are separated to maintain different environments, but each take a tensorflow.js model that comes from Teachable Machine and convert it to a list of outputs.

## Using endpoint

Each endpoint accepts POST requests in the format specifie below. Attach the model and dataset to your request body, and use the path to set the type and format.

## Run locally

```bash
docker compose up
```

## Supported Formats

### Image converter

Converts the image model to a range of formats

localhost url: `localhost:9002/convert/{type}/{format}/`

| arg       | type      |                                                                               |
| --------- | --------- | ----------------------------------------------------------------------------- |
| `model`   | .zip file | a file containing the tf.js model                                             |
| `dataset` | .zip file | a file containing representative dataset (only needed for EdgeTPU conversion) |
| `type`    | string    | `image` |
| `format`  | string    | `savedmodel` / `keras` / `tflite` / `tflite_quantized` / `edgetpu`            |

### Audio Converter

Converts the audio model to TFlite

localhost url: `localhost:9003/convert/{type}/{format}/`

| arg       | type      |                                                                               |
| --------- | --------- | ----------------------------------------------------------------------------- |
| `model`   | .zip file | a file containing the tf.js model                                             |
| `dataset` | .zip file | a file containing representative dataset (only needed for EdgeTPU conversion) |
| `type`    | string    |  `audio` |
| `format`  | string    | `tflite` |

### Tiny Converter

Used to convet the 96x96px grayscale model to TFLite and TFLite for Microcontrollers format

localhost url: `localhost:9001/convert/{type}/{format}/`

| arg       | type      |                                                                               |
| --------- | --------- | ----------------------------------------------------------------------------- |
| `model`   | .zip file | a file containing the tf.js model                                             |
| `dataset` | .zip file | a file containing representative dataset (only needed for EdgeTPU conversion) |
| `type`    | string    |  `tiny_image` |
| `format`  | string    | `tflite, tinyml` |

## Test

Uncomment lines 19-23 in `docker-compose.yml`

```yaml
## Uncomment these lines to enable tests
   networks:
     default:
         external:
             name: cloudbuild

```

Then run the cloudbuild testfile:
[Learn more about testing locally with cloudbuild here](https://cloud.google.com/build/docs/build-debug-locally)

```bash
cloud-build-local --config cloudbuild.test.yaml  --dryrun=false .
```