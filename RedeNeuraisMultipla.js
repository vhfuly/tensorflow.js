const tf = require('@tensorflow/tfjs');

async function run() {
  const x = tf.tensor([[1,2,3],[4,5,6],[7,8,9]],[3,3]);
  const y = tf.tensor([[6],[15],[24]]);
  const input = tf.tensor([[10,11,12], [13,14,15]],[2,3]);

  //treinamento
  const model = await trainLinearModel(x,y)
  let output = model.predict(input).dataSync();

  output = convertArray(output);
  let z = tf.tensor(output)

  console.log(`
    Regressão Linear Multipla com Rede Neural

    Treinamento: 

    ${x.toString()}

    ${y.toString()}

    Entrada: 

    ${input.toString()}

    Saida: 
    
    ${z.toString()}
  `)
}

const trainLinearModel = async (x, y) => {
  const model = tf.sequential();
  const inputLayer = tf.layers.dense({
    units: 1, 
    inputShape: [3],
  })
  model.add(inputLayer);
  model.compile({
    loss: 'meanSquaredError',//otimização de error
    optimizer: 'sgd', //função de otimização de pesos
  });

  //treinamento

  await model.fit(x, y, {epochs: 550}); //epochs quantidade de vezes para obter a resposta desejada
  return model
}

const convertArray = (array) => {
  let result = [];
  for(let i=0 ; i< array.length; i++) {
    result.push(Number(array[i].toFixed(0)))
  }
  return result;
} 

run();