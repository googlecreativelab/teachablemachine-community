# Teachable Machine Support Libraries (beta)

This repo contains support libraries for a new version of Teachable Machine (currently in beta). For more info or request to be a beta tester: [Teachable Machine](https://teachablemachine.withgoogle.com/io19).

## Model Libraries

| Library | Based on model  | Details                                                 | Install | CDN | 
|---------|-----------------|---------------------------------------------------------|---------|-----|
| [Image](./image/) | [MobileNet](https://github.com/tensorflow/tfjs-models/tree/master/mobilenet)       | Use a model trained to classify your own images         | `npm i @teachablemachine/image` | [![](https://data.jsdelivr.com/v1/package/npm/@teachablemachine/image/badge)](https://www.jsdelivr.com/package/npm/@teachablemachine/image) |
| [Audio](./audio/)   | [Speech Commands](https://github.com/tensorflow/tfjs-models/tree/master/speech-commands) | Use a model trained to classify your own audio snippets | Coming soon     |  | 
| [Pose](./pose/)   | [PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/posenet) | Use a model trained to classify body poses | `npm i @teachablemachine/pose`     | [![](https://data.jsdelivr.com/v1/package/npm/@teachablemachine/pose/badge)](https://www.jsdelivr.com/package/npm/@teachablemachine/pose) |

## Development

You must use a node version > 12.

## Disclaimer

This is an experiment, not an official Google product. We’ll do our best to support and maintain this experiment but your mileage may vary.

We encourage open sourcing projects as a way of learning from each other. Please respect our and other creators’ rights, including copyright and trademark rights when present, when sharing these works and creating derivative work. If you want more info on Google's policy, you can find that [here](https://www.google.com/permissions/).
