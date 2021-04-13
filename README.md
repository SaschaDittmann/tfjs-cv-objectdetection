[![Software License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![](https://ga4gh.datainsights.cloud/api?repo=tfjs-cv-objectdetection)](https://github.com/SaschaDittmann/gaforgithub)

# TensorFlow.js Example: Using an Azure Custom Vision Object Detection model to detect Logos in a web browser

This example shows you how to use a Machine Learning, which was created with the [Microsoft Azure Custom Vision](https://azure.microsoft.com/en-us/services/cognitive-services/custom-vision-service/) service, in a web browser application.

The [Azure Logo images](https://github.com/microsoft/AIVisualProvision/tree/master/Documents/Images/Training_DataSet) used in this example, are from the AI Vision Provision demo shown at the Microsoft Connect() event in 2018 and are not published with this repository.

## About

For more details about this project, please checkout my [YouTube Video](https://www.youtube.com/watch?v=7gOYpT732ow&list=PLZk8J6FocZbaClHkIPk4SWZHxn_9VArb5&index=2):

[![Watch the video](https://img.youtube.com/vi/7gOYpT732ow/maxresdefault.jpg)](https://www.youtube.com/watch?v=7gOYpT732ow&list=PLZk8J6FocZbaClHkIPk4SWZHxn_9VArb5&index=2)

## Setup 

Prepare the node environments:
```sh
$ npm install
# Or
$ yarn
```

Run the local web server script:
```sh
$ node server.js
```

## Demo

If you wan't, you can test the deployed application under under [https://tfjs-objectdetection.azureedge.net](https://tfjs-objectdetection.azureedge.net).
