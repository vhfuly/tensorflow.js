const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const format = require('date-fns/format')
const addDays = require('date-fns/addDays')
const model = tf.sequential();

const run = async () => {

    const X = [];
    const Y = [];

    let file1 = fs.readFileSync('bitcoin01.csv', { encoding: 'utf-8'});
    file1 = file1.toString().trim();

    const rows1 = file1.split('\n');
    for(let r = rows1.length -1; r >= 1; r--) {
        let cell = rows1[r].toString().split(';');

        const close = Number(cell[1]);
        const open = Number(cell[2]);
        const max = Number(cell[3]);
        const min = Number(cell[4]);
        
        X.push([[close], [open], [max] , [min]])
    }


    let file2 = fs.readFileSync('bitcoin02.csv', { encoding: 'utf-8'});
    file1 = file2.toString().trim();

    const rows2 = file2.split('\n');
    for(let r = rows2.length -1; r >= 1; r--) {
        let cell = rows2[r].toString().split(';');

        const close = Number(cell[1]);
        const open = Number(cell[2]);
        const max = Number(cell[3]);
        const min = Number(cell[4]);
        Y.push([[close], [open], [max] , [min]])
    }

    const x = tf.tensor(X);
    const y = tf.tensor(Y);

    const rnn = tf.layers.simpleRNN({units: 1, returnSequences: true, activation: 'linear'});

    const inputLayer = tf.input({shape: [X[0].length, 1]});
    rnn.apply(inputLayer);

    model.add(rnn);

    model.compile({loss: 'meanSquaredError', optimizer: tf.train.adam(0.001)});
    for(let i=1; i<= 20000; i++){

        await model.fit(x, y);
    }
    
};

const predict = async () => {

    let Z =[]

    let file = fs.readFileSync('bitcoin03.csv', { encoding: 'utf-8'});
    file = file.toString().trim();

    const rows = file.split('\n');
    for(let r = rows.length -1; r >= 1; r--) {
        let cell = rows[r].toString().split(';');

        const close = Number(cell[1]);
        const open = Number(cell[2]);
        const max = Number(cell[3]);
        const min = Number(cell[4]);
        Z.push([[close], [open], [max] , [min]])
    }

    const input = tf.tensor(Z)
    const output = await model.predict(input).abs().arraySync();


    const today = new Date();

    console.log(`   Data    - Fechamento -  Abertura  -   Máxima   -   Minima  `);
    console.log(`--------------------------------------------------------------`)
    for(let i = 0; i<output.length; i++) {
        let currentDate =  format(addDays(today, i),'dd/MM/yyyy')

        const close = parseFloat(output[i][0]).toFixed(1);
        const open = parseFloat(output[i][1]).toFixed(1);
        const max = parseFloat(output[i][2]).toFixed(1);
        const min = parseFloat(output[i][3]).toFixed(1);
  

        console.log(`${currentDate.padEnd(10, ' ')} - ${close.padEnd(10, ' ')} - ${open.padEnd(10, ' ')} - ${max.padEnd(10, ' ')} - ${min.padEnd(10, ' ')}`)
    }
}

(async() => {
    await run();
    await predict();
})();