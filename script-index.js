async function loadData(path) {
    const response = await fetch(path);
    return await response.json();
}

async function loadModelFromPath(path) {
    return await tf.loadLayersModel(path);
}

async function initAll() {
    rawData = await loadData('./data/rawData.json');
    noisyData = await loadData('./data/noisyData.json');
    trainSet = await loadData('./data/trainSet.json');
    testSet = await loadData('./data/testSet.json');

    plotData();

    const modelUnverrauscht = await loadModelFromPath('./models/unverrauscht/unverrauscht-model.json');
    modelUnverrauscht.compile({
        loss: 'meanSquaredError',
        optimizer: tf.train.adam(0.01)
    });

    const modelBestfit = await loadModelFromPath('./models/bestfit/bestfit-model.json');
    modelBestfit.compile({
        loss: 'meanSquaredError',
        optimizer: tf.train.adam(0.01)
    });

    const modelOverfit = await loadModelFromPath('./models/overfit/overfit-model.json');
    modelOverfit.compile({
        loss: 'meanSquaredError',
        optimizer: tf.train.adam(0.01)
    });

    await evaluateModel(modelUnverrauscht, 'unverrauscht');
    await evaluateModel(modelBestfit, 'bestfit');
    await evaluateModel(modelOverfit, 'overfit');
}

if (document.readyState !== "loading") {
    initAll();
} else {
    document.addEventListener("DOMContentLoaded", initAll);
}


// Bewertet das Modell auf Trainings- und Testdaten und zeigt Vorhersagen
async function evaluateModel(model, type) {
    const ds = type === 'unverrauscht' ? trainSet.raw : trainSet.noisy;
    const ts = type === 'unverrauscht' ? testSet.raw : testSet.noisy;

    const [trainLoss, trainPreds] = await testLossAndPreds(model, ds);
    const [testLoss, testPreds] = await testLossAndPreds(model, ts);

    // Visualisierung der Vorhersagen
    plotPredictions(type, ds, trainPreds, ts, testPreds);

    // Anzeige der Fehlerwerte
    displayLoss(type, trainLoss, testLoss);
}

// Berechnet den Fehler und gibt die Vorhersagen für gegebene Daten zurück
async function testLossAndPreds(model, data) {
    const xs = tf.tensor2d(data.map(d => [d.x]));
    const ys = tf.tensor2d(data.map(d => [d.y]));

    // Vorhersagen vom Modell berechnen
    const preds = model.predict(xs);

    // Fehler (Loss) berechnen
    const lossTensor = await model.evaluate(xs, ys);
    const loss = lossTensor.dataSync()[0];

    // Vorhersagewerte extrahieren
    const yPred = preds.dataSync();
    const predPoints = data.map((d, i) => ({ x: d.x, y: yPred[i] }));

    // Ressourcen aufräumen
    xs.dispose();
    ys.dispose();
    preds.dispose();
    lossTensor.dispose();

    return [loss, predPoints];
}

// Visualisiert die Daten und Modellvorhersagen mit Plotly
function plotData() {
    const layoutEins = { title: 'Datensätze ohne Rauschen', xaxis: { title: 'x' }, yaxis: { title: 'y' } };
    const layoutZwei = { title: 'Datensätze mit Rauschen', xaxis: { title: 'x' }, yaxis: { title: 'y' } };


    // Rohdaten plotten
    Plotly.newPlot('plot-data-clean', [
        { x: trainSet.raw.map(p => p.x), y: trainSet.raw.map(p => p.y), mode: 'markers', name: 'Train' },
        { x: testSet.raw.map(p => p.x), y: testSet.raw.map(p => p.y), mode: 'markers', name: 'Test' }
    ], layoutEins);

    // Verrauschte Daten plotten
    Plotly.newPlot('plot-data-noisy', [
        { x: trainSet.noisy.map(p => p.x), y: trainSet.noisy.map(p => p.y), mode: 'markers', name: 'Train' },
        { x: testSet.noisy.map(p => p.x), y: testSet.noisy.map(p => p.y), mode: 'markers', name: 'Test' }
    ], layoutZwei);
}

function plotPredictions(type, train, trainPred, test, testPred) {
    const layoutEins = { title: 'Trainingsdaten', xaxis: { title: 'x' }, yaxis: { title: 'y' } };
    const layoutZwei = { title: 'Testdaten', xaxis: { title: 'x' }, yaxis: { title: 'y' } };

    let trainId = `plot-predict-${type}-train`;
    let testId = `plot-predict-${type}-test`;

    // Sortiere Vorhersagen nach x
    trainPred.sort((a, b) => a.x - b.x);
    testPred.sort((a, b) => a.x - b.x);

    Plotly.newPlot(trainId, [
        { x: train.map(p => p.x), y: train.map(p => p.y), mode: 'markers', name: 'True' },
        { x: trainPred.map(p => p.x), y: trainPred.map(p => p.y), mode: 'lines', name: 'Predicted' }
    ], layoutEins);

    Plotly.newPlot(testId, [
        { x: test.map(p => p.x), y: test.map(p => p.y), mode: 'markers', name: 'True' },
        { x: testPred.map(p => p.x), y: testPred.map(p => p.y), mode: 'lines', name: 'Predicted' }
    ], layoutZwei);
}


// Zeigt die Fehlerwerte (Loss) im HTML-Dokument an
function displayLoss(type, trainLoss, testLoss) {
    const div = document.getElementById(`loss-values-${type}`);
    const html = `<h3 class="loss-header">Fehlerwerte</h3>
    <p class="loss-text">Train Loss (MSE): ${trainLoss.toFixed(4)}</p>
    <p class="loss-text">Test Loss (MSE): ${testLoss.toFixed(4)}</p>`;
    div.innerHTML = html;
}
