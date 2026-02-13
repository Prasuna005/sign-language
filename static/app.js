const socket = io();
const predEl = document.getElementById("pred");
const confEl = document.getElementById("conf");
const sentenceEl = document.getElementById("sentence");

let sentence = "";
let lastPrediction = null;
let stableSince = null;
const STABLE_TIME = 1500; // ms (1.5 seconds)

// Controls
document.getElementById("space").onclick = () => {
  sentence += " ";
  sentenceEl.textContent = sentence;
};
document.getElementById("del").onclick = () => {
  sentence = sentence.slice(0, -1);
  sentenceEl.textContent = sentence;
};
document.getElementById("clear").onclick = () => {
  sentence = "";
  sentenceEl.textContent = sentence;
};

// Listen for predictions from backend
socket.on("prediction", data => {
  const {label, conf} = data;
  predEl.textContent = `Prediction: ${label}`;
  confEl.textContent = `Confidence: ${(conf*100).toFixed(1)}%`;

  const now = Date.now();
  if (label === lastPrediction) {
    if (!stableSince) stableSince = now;
    if (now - stableSince >= STABLE_TIME) {
      sentence += label;
      sentenceEl.textContent = sentence;
      stableSince = null;
      lastPrediction = null; // reset so it won't repeat same char continuously
    }
  } else {
    lastPrediction = label;
    stableSince = now;
  }
});

