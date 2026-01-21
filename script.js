let data;

async function load() {
  const res = await fetch("results/metrics.json");
  data = await res.json();
  updateFromSlider();
}

function nearestIndex(arr, x) {
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < arr.length; i++) {
    const d = Math.abs(arr[i] - x);
    if (d < bestDist) { bestDist = d; best = i; }
  }
  return best;
}

function updateFromSlider() {
  const slider = document.getElementById("alpha");
  const alpha = parseFloat(slider.value);

  document.getElementById("alphaVal").textContent = alpha.toFixed(2);

  const i = nearestIndex(data.alphas, alpha);
  const a = data.alphas[i];

  document.getElementById("acc").textContent = (data.accuracy[i] * 100).toFixed(1) + "%";
  document.getElementById("steer").textContent = data.steering[i].toFixed(2);

  const ex = data.examples[String(a)] || data.examples[String(alpha)];
  if (ex) {
    document.getElementById("beforeTxt").textContent = ex.before;
    document.getElementById("afterTxt").textContent = ex.after;
  }
}

document.addEventListener("input", (e) => {
  if (e.target && e.target.id === "alpha" && data) updateFromSlider();
});

load();

