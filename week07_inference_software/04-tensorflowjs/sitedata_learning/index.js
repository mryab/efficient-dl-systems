let videostream;
let labelBox;
let mlModel;
let newClassifier;

let bottleButton;
let cupButton;

function setup() {
    console.log('Setup JS application');

    noCanvas();

    videostream = createCapture('video');
    videostream.play();

    labelBox = createElement('h2', 'Prediction');

    mlModel = ml5.featureExtractor('MobileNet', modelReady);

    newClassifier  = mlModel.classification(videostream);

    bottleButton = createButton('Bottle');
    bottleButton.mousePressed(function() {
      console.log("Added Bottle");
      newClassifier.addImage('Bottle');
    });

    cupButton = createButton('Cup');
    cupButton.mousePressed(function() {
      console.log("Added Cup");
      newClassifier.addImage('Cup');
    });

    trainButton = createButton('Train');
    trainButton.mousePressed(function() {
      console.log("Begin training");
      newClassifier.train(controlTraining);
    });
}

function controlTraining(loss) {
  if (loss) {
    console.log(loss);
  } else {
    newClassifier.classify(drawPrediction);
  }
}

function modelReady() {
    console.log('Model is ready to make predictions');
}

function drawPrediction(error, result) {
  if (!error) {
    prediction = result[0]['label'];
    probability = result[0]['confidence'];
    labelBox.html(prediction + " - " + probability);

    newClassifier.classify(drawPrediction);
  }
}
