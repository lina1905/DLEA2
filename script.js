// script.js
// Rohdaten und verrauschte Daten + Aufteilung in Trainings- und Testdaten
let rawData = [], noisyData = [], trainSet = {}, testSet = {};

// Funktion zum Erzeugen von Datenpunkten
function generateData() {
    // Die Ziel-Funktion, die wir lernen wollen (f(x))
    const f = x => 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
    const N = 100; // Anzahl der Punkte
    rawData = [], noisyData = [];

    for (let i = 0; i < N; i++) {
        const x = Math.random() * 4 - 2; // x-Werte im Bereich [-2, 2]
        const y = f(x); // y-Wert berechnen
        rawData.push({ x, y }); // saubere Daten speichern

        // Etwas Rauschen hinzufügen (wie in echten Messdaten)
        const noise = gaussianNoise(0, Math.sqrt(0.05));
        noisyData.push({ x, y: y + noise });
    }

    // Daten in Trainings- und Testsets aufteilen
    [trainSet.raw, testSet.raw] = splitData(rawData);
    [trainSet.noisy, testSet.noisy] = splitData(noisyData);

    // Daten visualisieren
    plotData();
}

// Erzeugt normalverteiltes Rauschen (Zufallswert mit Mittelwert und Standardabweichung)
function gaussianNoise(mean = 0, stddev = 1) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return stddev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) + mean;
}

// Teilt ein Array zufällig in zwei Hälften
function splitData(data) {
    const shuffled = [...data].sort(() => 0.5 - Math.random());
    const mid = Math.floor(shuffled.length / 2);
    return [shuffled.slice(0, mid), shuffled.slice(mid)];
}

// Trainingsfunktion für das Modell mit verschiedenen Einstellungen (über 'type')
async function trainModel(type) {
    // Auswahl: mit oder ohne Rauschen trainieren
    let data = type === 'unverrauscht' ? trainSet.raw : trainSet.noisy;

    // Wähle Anzahl der Trainingsdurchläufe je nach Typ
    const epochs = type === 'overfit' ? 300 : (type === 'bestfit' ? 50 : 100);
    const model = createModel();

    // Eingabedaten (x) und Ausgabedaten (y) in Tensoren umwandeln
    const xs = tf.tensor2d(data.map(d => [d.x]));
    const ys = tf.tensor2d(data.map(d => [d.y]));

    // Modell trainieren
    await model.fit(xs, ys, {
        batchSize: 32,
        epochs,
        shuffle: true, // zufällige Reihenfolge
        verbose: 0 // keine Ausgabe im Log
    });

    // Tensoren wieder freigeben
    xs.dispose();
    ys.dispose();

    // Modell bewerten (auf Trainings- und Testdaten)
    await evaluateModel(model, type);

    await plotLearnedFunction(model, type);
}

// Erstellt ein einfaches neuronales Netz mit 2 versteckten Schichten (je 100 Neuronen)
function createModel() {
    const model = tf.sequential();

    // Erste Schicht mit ReLU-Aktivierung
    model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));

    // Zweite Schicht (auch ReLU)
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));

    // Ausgangsschicht (1 Wert)
    model.add(tf.layers.dense({ units: 1 }));

    // Modell kompilieren: Fehlerfunktion & Optimierer festlegen
    model.compile({
        loss: 'meanSquaredError', // mittlerer quadratischer Fehler
        optimizer: tf.train.adam(0.01) // Adam ist ein moderner Lernalgorithmus
    });

    return model;
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
    const layout = { title: 'Datensätze', xaxis: { title: 'x' }, yaxis: { title: 'y' } };

    // Rohdaten plotten
    Plotly.newPlot('plot-data-clean', [
        { x: trainSet.raw.map(p => p.x), y: trainSet.raw.map(p => p.y), mode: 'markers', name: 'Train' },
        { x: testSet.raw.map(p => p.x), y: testSet.raw.map(p => p.y), mode: 'markers', name: 'Test' }
    ], layout);

    // Verrauschte Daten plotten
    Plotly.newPlot('plot-data-noisy', [
        { x: trainSet.noisy.map(p => p.x), y: trainSet.noisy.map(p => p.y), mode: 'markers', name: 'Train' },
        { x: testSet.noisy.map(p => p.x), y: testSet.noisy.map(p => p.y), mode: 'markers', name: 'Test' }
    ], layout);
}

// Zeichnet die Vorhersagen des Modells (Trainings- und Testdaten)
function plotPredictions(type, train, trainPred, test, testPred) {
    const layout = { xaxis: { title: 'x' }, yaxis: { title: 'y' } };
    let trainId = `plot-predict-${type}-train`;
    let testId = `plot-predict-${type}-test`;

    // Sortiere Vorhersagen nach x
    trainPred.sort((a, b) => a.x - b.x);
    testPred.sort((a, b) => a.x - b.x);

    Plotly.newPlot(trainId, [
        { x: train.map(p => p.x), y: train.map(p => p.y), mode: 'markers', name: 'True' },
        { x: trainPred.map(p => p.x), y: trainPred.map(p => p.y), mode: 'lines', name: 'Predicted' }
    ], layout);

    Plotly.newPlot(testId, [
        { x: test.map(p => p.x), y: test.map(p => p.y), mode: 'markers', name: 'True' },
        { x: testPred.map(p => p.x), y: testPred.map(p => p.y), mode: 'lines', name: 'Predicted' }
    ], layout);
}


async function plotLearnedFunction(model, type) {
    // x-Werte von -2 bis 2, gleichmäßig verteilt
    const xs = tf.linspace(-2, 2, 200).reshape([200, 1]);
    const preds = model.predict(xs);
    const yPred = await preds.data();

    // Umwandeln in Array von Punkten
    const predPoints = Array.from(xs.dataSync()).map((x, i) => ({
        x, y: yPred[i]
    }));

    // Zeichnen
    Plotly.newPlot(`plot-function-${type}`, [
        {
            x: predPoints.map(p => p.x),
            y: predPoints.map(p => p.y),
            mode: 'lines',
            name: 'Learned Function'
        }
    ], { title: `Gelernte Funktion (${type})`, xaxis: { title: 'x' }, yaxis: { title: 'y' } });

    // Aufräumen
    xs.dispose();
    preds.dispose();
}


// Zeigt die Fehlerwerte (Loss) im HTML-Dokument an
function displayLoss(type, trainLoss, testLoss) {
    const div = document.getElementById('loss-values');
    const html = `<h3>${type.toUpperCase()} Modell</h3>
    <p>Train Loss (MSE): ${trainLoss.toFixed(4)}</p>
    <p>Test Loss (MSE): ${testLoss.toFixed(4)}</p>`;
    div.innerHTML += html;
}

function saveDataset(name, data) {
    localStorage.setItem(name, JSON.stringify(data));
}

function loadDataset(name) {
    const data = localStorage.getItem(name);
    return data ? JSON.parse(data) : null;
}

async function saveModel(model, name) {
    if (model) {
        await model.save(`indexeddb://${name}`);
        alert('Modell gespeichert!');
    }
}
async function loadAndShowModel(name) {
    const model = await tf.loadLayersModel(`indexeddb://${name}`);
    // Zeige nach dem Laden die Vorhersagen an
    await evaluateModel(model, name);
}

