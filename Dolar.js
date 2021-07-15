const tf = require('@tensorflow/tfjs');
const fs = require('fs');

let file = fs.readFileSync('csv/cotacao-do-dolar.csv', {encoding: 'utf-8'});
file = file.toString().trim();

const rows = file.split('\n'); //windows \r\n;

let X = [];
let Y = [];
let qtdRows = 0;

async function run() {
  for(let r = 1; r< rows.length; r++) {
    let cells1 = [];
    if(qtdRows == (rows.length-2)) cells1 = ['31.12.2018', 3.88813, 3.88813, 3.88813, 3.88813];
    else cells1 = rows[r+1].split(';');

    const cells2 = rows[r].split(';');

    const closureX = Number(cells1[1]);
    const openingX = Number(cells1[2]);
    const maxX = Number(cells1[3]);
    const minX = Number(cells1[4]);

    X.push([closureX, openingX, maxX, minX])

    const closureY = Number(cells2[1]);
    const openingY = Number(cells2[2]);
    const maxY = Number(cells2[3]);
    const minY = Number(cells2[4]);

    Y.push([closureY, openingY, maxY, minY])
    qtdRows++;
  }

  // const arrInput = [[3.9285, 3.9708, 3.9781, 3.9251]] //08-05/2019
  const arrInput = [[3.9572, 3.9470, 3.9736, 3.9363]] //09-05/2019
  const input = tf.tensor(arrInput, [1,4])
  const model = await trainLinearModel(X, Y);
  let output = model.predict(input).dataSync();

  console.log(`
    Preço das cotações

    Fechamento :
    R$ ${Number(output[0].toFixed(4))}

    Abertura :
    R$ ${Number(output[1].toFixed(4))}

    Máxima :
    R$ ${Number(output[2].toFixed(4))}

    Mínima :
    R$ ${Number(output[3].toFixed(4))}

  `)
}


const trainLinearModel = async (X, Y) => {
  const model = tf.sequential();
  const inputLayer = tf.layers.dense({
    units: 4, // resposta com 4 dimensões
    inputShape: [4],
  })
  model.add(inputLayer);
  model.compile({
    loss: 'meanSquaredError',//otimização de error
    optimizer: 'sgd', //função de otimização de pesos
  });

  const x  = tf.tensor(X, [qtdRows, 4 ]);
  const y  = tf.tensor(Y, [qtdRows, 4 ]);

  //treinamento

  await model.fit(x, y, {
    epochs: 500,
  }); //epochs quantidade de vezes para obter a resposta desejada
  return model
}

run();