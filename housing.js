const tf = require('@tensorflow/tfjs');
const tfvis = require('@tensorflow/tfjs-vis');

const csv = require('csvtojson');
const csvString = require('./housingData');
const regression = require('regression');

const prepareData = async () => {
  data = await csv().fromString(csvString);
  return data;
  
};

const creatingFeatures = (data) => {
  let features = []
  data.forEach(parameters => {
    features.push([
      parameters['OverallQual'] ? Number(parameters['OverallQual']) : 0,
      parameters['GrLivArea'] ? Number(parameters['GrLivArea']) : 0,
      parameters['GarageCars'] ? Number(parameters['GarageCars']) : 0,
      parameters['TotalBsmtSF'] ? Number(parameters['TotalBsmtSF']) : 0,
      parameters['FullBath'] ? Number(parameters['FullBath']) : 0,
      parameters['YearBuilt'] ? Number(parameters['YearBuilt']) : 0,
    ])
  } 
  )
  return features
};

const creatingLabels = (data) => {
  let labels = []
  data.forEach(parameters => {
    labels.push([parameters['SalePrice'] ? Number(parameters['SalePrice']) : 0])
  } 
  )
  return labels
};


const normalize = tensor =>
  tf.div(
    tf.sub(tensor, tf.min(tensor)),
    tf.sub(tf.max(tensor), tf.min(tensor))
  );

const createDataSets = (data) => {

  const features = creatingFeatures(data);
  const labels = creatingLabels(data)

  const qtd_linhas = data.length
  const qtd_linhas_treino = Math.round(0.90 * qtd_linhas)
  const qtd_linhas_teste= qtd_linhas - qtd_linhas_treino

  const X_train = features.slice(0, qtd_linhas_treino)
  const X_test = features.slice(qtd_linhas_treino, qtd_linhas_treino + qtd_linhas_teste)
  const y_train = labels.slice(0, qtd_linhas_treino)
  const y_test = labels.slice(qtd_linhas_treino, qtd_linhas_treino + qtd_linhas_teste)
  const xTrain = normalize(tf.tensor(X_train,[X_train.length, X_train[0].length]));
  const xTest = normalize(tf.tensor(X_test,[X_test.length, X_test[0].length]));

  const yTrain = tf.tensor(y_train,[y_train.length, y_train[0].length]);
  const yTest = tf.tensor(y_test,[y_test.length, y_test[0].length]);
  return [xTrain, xTest, yTrain, yTest];
};

// const trainLinearModel = async (xTrain, yTrain) => {
//   const model = tf.sequential();

//   model.add(
//     tf.layers.dense({
//       inputShape: [6],
//       units: 1,
//     })
//   );

//   model.compile({
//     optimizer: tf.train.sgd(0.0001),
//     loss: "meanSquaredError",
//   });

//   await model.fit(xTrain, yTrain, {
//     epochs: 500,
 
//   });

//   return model;
// };

const trainLinearModel = async (X, Y) => {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [X.shape[1]],
      units: 1,
    })
  );

  model.compile({
    optimizer: tf.train.sgd(0.0001),
    loss: tf.metrics.meanSquaredError,
    metrics: tf.metrics.meanSquaredError
  });

  await model.fit(X, Y, {
    batchSize: 32,
    epochs: 100,
    shuffle: true,
    validationSplit: 0.1,
  });

  return model;
}

const run = async () => {
  const data = await prepareData();
  createDataSets(data)

  const [xTrain, xTest, yTrain, yTest ] = createDataSets(data);
  const linearModel = await trainLinearModel(xTrain, yTrain);

  const trueValues = yTest.dataSync();
  const lmPreds = linearModel.predict(xTest).dataSync();
  const arr = renderPredictions(trueValues,lmPreds)
  const result = regression.linear(arr)
  console.log(result);
  console.log(`${result.r2*100} %`);

};

const renderPredictions = (trueValues, slmPredictions) => {
  const arr =[]
  for(let i = 0; i<= trueValues.length -1; i++){
   arr.push([trueValues[i], slmPredictions[i]])
  }
  return arr
}

run();