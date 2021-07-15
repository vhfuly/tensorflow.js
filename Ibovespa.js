const tf = require('@tensorflow/tfjs');
const fs = require('fs');

let file = fs.readFileSync('csv/ibovespa.csv', {encoding: 'utf-8'});
file = file.toString().trim();

const rows = file.split('\n');

let X = [];
let Y = [];
let qtdRows = 0;

async function run() {
  for(let r = 1; r< rows.length; r++) {
    let cells1 = [];
    if(qtdRows == (rows.length-2)) cells1 = ['05/07/2019',103628,104089,0,"0,44",102622,104176,"15,15B"];
    else cells1 = rows[r+1].split(',');

    const cells2 = rows[r].split(',');

    const openingX = Number(cells1[1]);
    const closureX = Number(cells1[2]);
    const minX = Number(cells1[5]);
    const maxX = Number(cells1[6]);

    X.push([openingX, closureX, minX, maxX])
    
    const closureY = Number(cells2[2]);
 
 
    Y.push([closureY])
    qtdRows++;
  }

  const arrInput = [[126919, 125095, 125094, 127249]] //06/07/2021
  const input = tf.tensor(arrInput, [1,4])
  const model = await trainLinearModel(X, Y);
  let output = model.predict(input).dataSync();
  const linear = await regressaoLinear(X, Y, arrInput)
  console.log(`
    Preço das cotações

    Valor de fechamento do dia 07/07/2021:
    R$ 127019

    Fechamento Rede Neural:
    R$ ${Number(output[0].toFixed(0))}

    Fechamento Regressão Linear:
    R$ ${Number(linear[0].toFixed(0))}
  `)
}


const trainLinearModel = async (X, Y) => {
  const model = tf.sequential();
  const inputLayer = tf.layers.dense({
    units: 1, // resposta com 4 dimensões
    inputShape: [4],
  })
  model.add(inputLayer);
  const learningRate = 0.0000000000001;//(O numero de zeros depois do ponto, será igual a o dobro de dezenas do resultado)
  model.compile({
    loss: 'meanSquaredError',//otimização de error
    optimizer: tf.train.sgd(learningRate), //função de otimização de pesos;
  });

  const x  = tf.tensor(X, [qtdRows, 4 ]);
  const y  = tf.tensor(Y, [qtdRows, 1 ]);

  //treinamento

  await model.fit(x, y, {epochs: 500}); //epochs quantidade de vezes para obter a resposta desejada
  return model
}

async function regressaoLinear(arrX, arrY, input) {
  const testX = await newArray(arrX)
  let x = tf.tensor(testX);
  let y = tf.tensor(arrY);
  let z = tf.tensor(input);

  //regressão linear
  let resultado1 = x.sum().mul(y.sum()).div(x.size); // sum = soma  mul = multiplicação div= divisão 
  let resultado2 = x.sum().mul(x.sum()).div(x.size);
  let resultado3 = x.mul(y).sum().sub(resultado1);
  let resultado4 = resultado3.div(x.square().sum().sub(resultado2));
  let resultado5 = y.mean().sub(resultado4.mul(x.mean()));

  let tensor = resultado4.mul(z.mean()).add(resultado5);
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

run();