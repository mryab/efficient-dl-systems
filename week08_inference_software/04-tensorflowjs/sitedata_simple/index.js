let videostream;
let labelBox;
let mlModel;
let classifier;

function setup() {
    console.log('Setup JS application');

    noCanvas();
    videostream = createCapture('video');

    labelBox = createElement('h2', 'Prediction');
    mlModel = ml5.imageClassifier('MobileNet', videostream, modelReady);
}

function modelReady() {
    console.log('Model is ready to make predictions');
    mlModel.predict(drawPrediction)
}

function drawPrediction(error, result) {
  if (!error) {
    prediction = result[0]['label'];
    probability = result[0]['confidence'];
    labelBox.html(prediction + " - " + probability);

    mlModel.predict(drawPrediction);
  }
}
