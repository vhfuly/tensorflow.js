const tf = require('@tensorflow/tfjs');

const tensorX = tf.tensor([1,2,3,4,5,6,7,8,9])
const tensorY = tf.tensor([9,18,27,36])

let vetorX = tensorToArray(tensorX)
let vetorY = tensorToArray(tensorY)

let tamX = vetorX.length;
let tamY = vetorY.length;

let tempX = vetorX.slice(0, tamY);
let tempY = vetorY;

let dif = tamX -tamY;

if(dif > 0) {
  let regressao =[];
  for(let i=0 ; i<dif; i++) {
    let temp = regressaoLinear(tempX, tempY, vetorX[tamY + i] );
    regressao.push(temp);
  }
  let novoY = tempY.concat(regressao);
  let tensorZ = tf.tensor(novoY);

  console.log(`
    Regressão Linear simples
    
    Antes : 
    ${tensorX.toString()}
    ${tensorY.toString()}

    Depois : 
    ${tensorZ.toString()}
  `)
}

function tensorToArray(tensor) {
  let array = [];
  let strTensor = tensor.toString().replace('Tensor', '').trim();
  eval('array = ' + strTensor);

  return array;
}

function arrayToTensor(array) {
  let tensor = tf.tensor(array);
  return tensor;
}


function regressaoLinear(arrX, arrY, p) {
  let x = arrayToTensor(arrX);
  let y = arrayToTensor(arrY);

  //regressão linear
  let resultado1 = x.sum().mul(y.sum()).div(x.size); // sum = soma  mul = multiplicação div= divisão 
  let resultado2 = x.sum().mul(x.sum()).div(x.size);
  let resultado3 = x.mul(y).sum().sub(resultado1);
  let resultado4 = resultado3.div(x.square().sum().sub(resultado2));
  let resultado5 = y.mean().sub(resultado4.mul(x.mean()));

  let tensor = resultado4.mul(p).add(resultado5);
  let array = tensorToArray(tensor);

  return array;
}