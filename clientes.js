const tf = require('@tensorflow/tfjs');

const regression = require('regression');

const csv = require('csvtojson');
const csvString = require('./Churn');


const prepareData = async () => {
  data = await csv().fromString(csvString);
  return data;
  
};

const creatingFeatures = (data) => {
  let features = []
  data.forEach(parameters => {
    features.push([
      parameters.gender ? Number(parameters.gender): 0,  
      parameters.Partner ? Number(parameters.Partner): 0,  
      parameters.Dependents ? Number(parameters.Dependents): 0,  
      parameters.PhoneService ? Number(parameters.PhoneService): 0,   
      parameters.MultipleLines ? Number(parameters.MultipleLines): 0,   
      parameters.OnlineSecurity ? Number(parameters.OnlineSecurity): 0,   
      parameters.OnlineBackup ? Number(parameters.OnlineBackup): 0,   
      parameters.DeviceProtection ? Number(parameters.DeviceProtection): 0,   
      parameters.TechSupport ? Number(parameters.TechSupport): 0,   
      parameters.StreamingTV ? Number(parameters.StreamingTV): 0,   
      parameters.PaperlessBilling ? Number(parameters.PaperlessBilling): 0,   
      parameters.PaymentMethod ? Number(parameters.PaymentMethod): 0, 
      parameters.Contract ? Number(parameters.Contract): 0,
      parameters.InternetService ? Number(parameters.InternetService): 0,   
      parameters.InternetService ? Number(parameters.InternetService): 0,
      parameters.StreamingMovies ? Number(parameters.StreamingMovies): 0,   
      parameters.SeniorCitizen ? Number(parameters.SeniorCitizen): 0,
      parameters.tenure ? Number(parameters.tenure): 0, 
      parameters.MonthlyCharges ? Number(parameters.MonthlyCharges): 0, 
      parameters.TotalCharges ? Number(parameters.TotalCharges): 0,
    ])
  } 
  )
  return features
};

const creatingLabels = (data) => {
  let labels = []
  data.forEach(parameters => {
    labels.push([parameters['Churn'] ? Number(parameters['Churn']) : 0])
  } 
  )
  return labels
};


const createDataSets = (data) => {

  const features = creatingFeatures(data);
  const labels = creatingLabels(data)
 console.log(features)
  const qtd_linhas = data.length
  const qtd_linhas_treino = Math.round(0.90 * qtd_linhas)
  const qtd_linhas_teste= qtd_linhas - qtd_linhas_treino

  const X_train = features.slice(0, qtd_linhas_treino)
  const X_test = features.slice(qtd_linhas_treino, qtd_linhas_treino + qtd_linhas_teste)
  const y_train = labels.slice(0, qtd_linhas_treino)
  const y_test = labels.slice(qtd_linhas_treino, qtd_linhas_treino + qtd_linhas_teste)
  const xTrain = tf.tensor(X_train,[X_train.length, X_train[0].length]);
  const xTest = tf.tensor(X_test,[X_test.length, X_test[0].length]);

  const yTrain = tf.tensor(y_train,[y_train.length, y_train[0].length]);
  const yTest = tf.tensor(y_test,[y_test.length, y_test[0].length]);
  return [xTrain, xTest, yTrain, yTest];
};
const trainLinearModel = async (X, Y) => {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [X.shape[1]],
      units: 1,
      activation: 'sigmoid',
    })
  );

  model.compile({
    optimizer: tf.train.adam(0.0001),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  await model.fit(X, Y, {
    epochs: 100,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
       console.log(`
       epoch: ${epoch}/100 
       logs: ${logs.acc}`);
      }
    }
  });

  return model;
}

const run = async () => {
  let data = await prepareData();
  console.log(data.length)
  data = data.filter(value => {
    if(Number(value.TotalCharges)) return value
  })
 
  data = data.map(value => {
    value.gender  === 'Female' ? value.gender = 1 : value.gender = 0
    value.Partner  === 'Yes' ? value.Partner = 1 : value.Partner = 0
    value.Dependents  === 'Yes' ? value.Dependents = 1 : value.Dependents = 0
    value.PhoneService  === 'Yes' ? value.PhoneService = 1 : value.PhoneService = 0 
    value.MultipleLines  === 'Yes' ? value.MultipleLines = 1 : value.MultipleLines = 0 
    value.OnlineSecurity  === 'Yes' ? value.OnlineSecurity = 1 : value.OnlineSecurity = 0 
    value.OnlineBackup  === 'Yes' ? value.OnlineBackup = 1 : value.OnlineBackup = 0 
    value.DeviceProtection  === 'Yes' ? value.DeviceProtection = 1 : value.DeviceProtection = 0 
    value.TechSupport  === 'Yes' ? value.TechSupport = 1 : value.TechSupport = 0 
    value.StreamingTV  === 'Yes' ? value.StreamingTV = 1 : value.StreamingTV = 0 
    value.PaperlessBilling  === 'Yes' ? value.PaperlessBilling = 1 : value.PaperlessBilling = 0 
    value.PaymentMethod  === 'Credit card (automatic)' ? value.PaymentMethod = 1 : value.PaymentMethod = 0 
    value.Churn  === 'Yes' ? value.Churn = 1 : value.Churn = 0 
    value.Contract  === 'Month-to-month' ? value.Contract = 1 : value.Contract = 0
    value.InternetService  === 'DSL' ? value.InternetService_DSL = 1 : value.InternetService_DSL = 0 
    value.InternetService  === 'Fiber optic' ? value.InternetService_Fiber_optic = 1 : value.InternetService_Fiber_optic = 0 
    value.StreamingMovies  === 'Yes' ? value.StreamingMovies = 1 : value.StreamingMovies = 0 
    delete value.customerID
    delete value.InternetService
    value.SeniorCitizen ? value.SeniorCitizen = Number(value.SeniorCitizen) : value.SeniorCitizen = 0 
    value.tenure ? value.tenure = Number(value.tenure) : value.tenure = 0 
    value.MonthlyCharges ? value.MonthlyCharges = Number(value.MonthlyCharges) : value.MonthlyCharges = 0 
    value.TotalCharges ? value.TotalCharges = Number(value.TotalCharges) : value.TotalCharges = 0 
    return value
  })




  const [xTrain, xTest, yTrain, yTest ] = createDataSets(data);
  const linearModel = await trainLinearModel(xTrain, yTrain);

  const trueValues = yTest.dataSync();
  const lmPreds = linearModel.predict(xTest).dataSync();
  renderPredictions(trueValues,lmPreds)


};

const renderPredictions = (trueValues, slmPredictions) => {
  let acc = 0 
  let loss = 0
  for(let i = 0; i<= trueValues.length -1; i++){
    trueValues[i]===Math.round(Number(slmPredictions[i]))? acc++ : loss++;
  }
 console.log(`
 Acertos : ${acc}
 Erros: ${loss}
 
 Porcentagem de previsões corretas: ${(acc*100/trueValues.length).toFixed(2)}%`)
}

run();