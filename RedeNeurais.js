const tf = require('@tensorflow/tfjs');

async function run() {
  const x = tf.tensor([1,2,3,4],[4,1]);
  const y = tf.tensor([[9],[18],[27],[36]]);
  const input = tf.tensor([5,6,7],[3,1]);

  //treinamento
  const model = await trainLinearModel(x,y)
  let output = model.predict(input).dataSync();

  output = convertArray(output);
  let z = tf.tensor(output)

  console.log(`
    Regressão Linear Simples com Rede Neural

    Treinamento: 

    ${x.flatten().toString()}

    ${y.flatten().toString()}

    Entrada: 

    ${input.flatten().toString()}

    Saida: 
    
    ${z.toString()}
  `)
}

const trainLinearModel = async (x, y) => {
  const model = tf.sequential();
  model.add(tf.layers.dense({ 
    units: 1, // tamanho do tensor de resposta 
    inputShape: [1] // quantidade de colunas da camada 
  }));
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
    result.push(Math.ceil(array[i]))
  }
  return result;
} 

run();