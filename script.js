// script.js

let rawData = [], noisyData = [], trainSet = {}, testSet = {};

function generateData() {
    const f = x => 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
    const N = 100;
    rawData = [], noisyData = [];

    for (let i = 0; i < N; i++) {
        const x = Math.random() * 4 - 2; // [-2, 2]
        const y = f(x);
        rawData.push({ x, y });
        const noise = gaussianNoise(0, Math.sqrt(0.05));
        noisyData.push({ x, y: y + noise });
    }

    [trainSet.raw, testSet.raw] = splitData(rawData);
    [trainSet.noisy, testSet.noisy] = splitData(noisyData);

    plotData();
}

function gaussianNoise(mean = 0, stddev = 1) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return stddev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) + mean;
}

function splitData(data) {
    const shuffled = [...data].sort(() => 0.5 - Math.random());
    const mid = Math.floor(shuffled.length / 2);
    return [shuffled.slice(0, mid), shuffled.slice(mid)];
}

async function trainModel(type) {
    let data = type === 'unverrauscht' ? trainSet.raw : trainSet.noisy;
    const epochs = type === 'overfit' ? 300 : (type === 'bestfit' ? 50 : 100);
    const model = createModel();
    const xs = tf.tensor2d(data.map(d => [d.x]));
    const ys = tf.tensor2d(data.map(d => [d.y]));

    await model.fit(xs, ys, {
        batchSize: 32,
        epochs,
        shuffle: true,
        verbose: 0
    });

    xs.dispose();
    ys.dispose();

    await evaluateModel(model, type);
}

function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1 }));

    model.compile({
        loss: 'meanSquaredError',
        optimizer: tf.train.adam(0.01)
    });

    return model;
}

async function evaluateModel(model, type) {
    const ds = type === 'unverrauscht' ? trainSet.raw : trainSet.noisy;
    const ts = type === 'unverrauscht' ? testSet.raw : testSet.noisy;

    const [trainLoss, trainPreds] = await testLossAndPreds(model, ds);
    const [testLoss, testPreds] = await testLossAndPreds(model, ts);

    plotPredictions(type, ds, trainPreds, ts, testPreds);
    displayLoss(type, trainLoss, testLoss);
}

async function testLossAndPreds(model, data) {
    const xs = tf.tensor2d(data.map(d => [d.x]));
    const ys = tf.tensor2d(data.map(d => [d.y]));

    const preds = model.predict(xs);
    const lossTensor = await model.evaluate(xs, ys);
    const loss = lossTensor.dataSync()[0];

    const yPred = preds.dataSync();
    const predPoints = data.map((d, i) => ({ x: d.x, y: yPred[i] }));

    // Speicher freigeben
    xs.dispose();
    ys.dispose();
    preds.dispose();
    lossTensor.dispose();

    return [loss, predPoints];
}


function plotData() {
    const layout = { title: 'Datensätze', xaxis: { title: 'x' }, yaxis: { title: 'y' } };
    Plotly.newPlot('plot-data-clean', [
        { x: trainSet.raw.map(p => p.x), y: trainSet.raw.map(p => p.y), mode: 'markers', name: 'Train' },
        { x: testSet.raw.map(p => p.x), y: testSet.raw.map(p => p.y), mode: 'markers', name: 'Test' }
    ], layout);
    Plotly.newPlot('plot-data-noisy', [
        { x: trainSet.noisy.map(p => p.x), y: trainSet.noisy.map(p => p.y), mode: 'markers', name: 'Train' },
        { x: testSet.noisy.map(p => p.x), y: testSet.noisy.map(p => p.y), mode: 'markers', name: 'Test' }
    ], layout);
}

function plotPredictions(type, train, trainPred, test, testPred) {
    const layout = { xaxis: { title: 'x' }, yaxis: { title: 'y' } };
    let trainId = `plot-predict-${type}-train`;
    let testId = `plot-predict-${type}-test`;

    Plotly.newPlot(trainId, [
        { x: train.map(p => p.x), y: train.map(p => p.y), mode: 'markers', name: 'True' },
        { x: trainPred.map(p => p.x), y: trainPred.map(p => p.y), mode: 'lines', name: 'Predicted' }
    ], layout);

    Plotly.newPlot(testId, [
        { x: test.map(p => p.x), y: test.map(p => p.y), mode: 'markers', name: 'True' },
        { x: testPred.map(p => p.x), y: testPred.map(p => p.y), mode: 'lines', name: 'Predicted' }
    ], layout);
}

function displayLoss(type, trainLoss, testLoss) {
    const div = document.getElementById('loss-values');
    const html = `<h3>${type.toUpperCase()} Modell</h3>
    <p>Train Loss (MSE): ${trainLoss.toFixed(4)}</p>
    <p>Test Loss (MSE): ${testLoss.toFixed(4)}</p>`;
    div.innerHTML += html;
}
