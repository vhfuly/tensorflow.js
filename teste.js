const tf = require('@tensorflow/tfjs');
const csv = require('csvtojson');
const csvString = require('./ibovespa');
const regression = require('regression');


const parseCsvToJson = async () => {
  data = await csv().fromString(csvString);
  return data;
};

const manipulatingTheVolume = (data) => {
  data.map(parameter => {
    parameter.VOLUME = parameter.VOLUME.replace(',', '.')
    parameter.VOLUME = parameter.VOLUME.replace('B', '')
  }
  )
};

const creatingFeatures = (data) => {
  let features = []
  data.forEach(parameters => {
    features.push([
      Number(parameters.ABERTURA),
      Number(parameters['MÁXIMO']),
      Number(parameters['MÍNIMO']),
      Number(parameters.VOLUME),
    ])
  } 
  )
  return features
};

const creatingLabels = (data) => {
  let labels = []
  data.forEach(parameters => {
    labels.push([parameters.FECHAMENTO])
  } 
  )
  return labels
};

const passToNumber = (data) => {
  return data.map(number => {
    return Number(number)
  } 
  )
};

const normalize = tensor =>
  tf.div(
    tf.sub(tensor, tf.min(tensor)),
    tf.sub(tf.max(tensor), tf.min(tensor))
  );

  const trainLinearModel = async (xTrain, yTrain) => {
    const model = tf.sequential();
  
    model.add(
      tf.layers.dense({
        inputShape: [xTrain.shape[1]],
        units: xTrain.shape[1],
        activation: "sigmoid",
      })
    );
    model.add(tf.layers.dense({ 
      units: 1,
    }));
  
    model.compile({
      optimizer: tf.train.sgd(0.0001),
      loss: "meanSquaredError",
      metrics: [tf.metrics.meanAbsoluteError]
    });
  
    const trainLogs = [];
      await model.fit(xTrain, yTrain, {
        batchSize: 32,
        epochs: 1500,
        shuffle: true,
        validationSplit: 0.1,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            trainLogs.push(logs);
          }
        }
      });
    return model;
  };


const run = async () => {
  let data = await parseCsvToJson();
  data = data.reverse();

  const qtd_linhas = data.length
  const qtd_linhas_treino = Math.round(0.70 * qtd_linhas)
  const qtd_linhas_teste= qtd_linhas - qtd_linhas_treino
  const qtd_linhas_validacao = qtd_linhas -2

  console.log( `
    linhas treino= 0:${qtd_linhas_treino}
    linhas teste= ${qtd_linhas_treino}:${qtd_linhas_treino + qtd_linhas_teste -2}
    linhas validação= ${qtd_linhas_validacao}`
  );
  manipulatingTheVolume(data);

  const features = creatingFeatures(data)
  const labels = creatingLabels(data)

  const X_train = features.slice(0, qtd_linhas_treino)
  const X_test = features.slice(qtd_linhas_treino+1, qtd_linhas_treino + qtd_linhas_teste -2)

  const y_train = labels.slice(1, qtd_linhas_treino+1)
  const y_test = labels.slice(qtd_linhas_treino+2, qtd_linhas_treino + qtd_linhas_teste -1)
  console.log(X_train.length, y_train.length )
  console.log(X_test.length, y_test.length )

  const y_test_number = passToNumber(y_test)
  const y_train_number = passToNumber(y_train)

  // const X_train_scale = normalize(X_train)
  // const X_test_scale  = normalize(X_test) 
  const test = await regressaoLinear(X_train, y_train_number, [104090,103988,104679, 12.63])
  console.log(test)
  // console.log('shape:', X_train_scale.shape);
  // console.log('shape:', X_test_scale.shape);
  // tf.tensor2d(X_train, [X_train.length, 4])
  // const linearModel = await trainLinearModel(X_train_scale, tf.tensor(y_train_number,[y_train_number.length, 1]));
  

  // const slmPreds = linearModel.predict(tf.tensor(X_test, [X_test.length, 4])).dataSync();
  // const trueValues =  tf.tensor1d(y_test_number).dataSync();
  // // console.log(slmPreds);
  // // console.log(trueValues);
  // const arr = renderPredictions(trueValues,X_test_scale.dataSync())
  // const result = regression.linear(arr)
  // console.log(X_test)
  // const predcti = linearModel.predict(tf.tensor2d(features[349], [1, 4])).dataSync();
  // console.log(predcti)
  // console.log(labels[350])



}

async function regressaoLinear(arrX, arrY, input) {
  const X = await newArray(arrX)
  const P = (input[0] + input[1] + input[2]) /3
  let x = tf.tensor(X);
  let y = tf.tensor(arrY);

  //regressão linear
  let resultado1 = x.sum().mul(y.sum()).div(x.size); // sum = soma  mul = multiplicação div= divisão 
  let resultado2 = x.sum().mul(x.sum()).div(x.size);
  let resultado3 = x.mul(y).sum().sub(resultado1);
  let resultado4 = resultado3.div(x.square().sum().sub(resultado2));
  let resultado5 = y.mean().sub(resultado4.mul(x.mean()));

  let tensor = resultado4.mul(P).add(resultado5);
  let array = tensor.dataSync();

  return array;
}

const newArray = async (arrX) => {
  const X = []
  for (let i= 0; i< arrX.length ; i++) {
    const result = ( arrX[i][0] + arrX[i][1] + arrX[i][2] ) /3
    X.push([result])
  }
  return X;
}

const renderPredictions = (trueValues, slmPredictions) => {
  const arr =[]
  for(let i = 0; i<= trueValues.length -1; i++){
   arr.push([trueValues[i], slmPredictions[i]])
  }
  return arr
}
run();
