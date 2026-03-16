const canvas = document.getElementById("plot");
const context = canvas.getContext("2d");
const nextButton = document.getElementById("nextButton");
const resetButton = document.getElementById("resetButton");

const GRID_SIZE = 320;
const NOISE_VARIANCE = 1e-4;
const SIGNAL_VARIANCE = 1;
const LENGTH_SCALE = 0.11;
const EXPLORATION_WEIGHT = 2.2;
const JITTER = 1e-8;

const state = {
  objective: null,
  observations: [],
  pendingX: 0.5,
  grid: [],
  posterior: [],
  yMin: -1,
  yMax: 1,
};

function createGrid() {
  return Array.from({ length: GRID_SIZE }, (_, index) => index / (GRID_SIZE - 1));
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function kernel(a, b) {
  const distance = a - b;
  return SIGNAL_VARIANCE * Math.exp(-(distance * distance) / (2 * LENGTH_SCALE * LENGTH_SCALE));
}

function createRandomObjective() {
  const waves = Array.from({ length: 3 }, () => ({
    amplitude: 0.3 + Math.random() * 0.55,
    frequency: 0.7 + Math.random() * 3.8,
    phase: Math.random() * Math.PI * 2,
  }));
  const bumps = Array.from({ length: 4 }, () => ({
    amplitude: -0.9 + Math.random() * 1.8,
    center: 0.08 + Math.random() * 0.84,
    width: 0.04 + Math.random() * 0.16,
  }));

  return (x) => {
    const waveValue = waves.reduce((sum, wave) => {
      return sum + wave.amplitude * Math.sin((x * wave.frequency + wave.phase) * Math.PI * 2);
    }, 0);
    const bumpValue = bumps.reduce((sum, bump) => {
      const distance = x - bump.center;
      return sum + bump.amplitude * Math.exp(-(distance * distance) / (2 * bump.width * bump.width));
    }, 0);

    return (waveValue * 0.42 + bumpValue * 0.95) * 0.9;
  };
}

function cholesky(matrix) {
  const size = matrix.length;
  const lower = Array.from({ length: size }, () => Array(size).fill(0));

  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column <= row; column += 1) {
      let sum = matrix[row][column];
      for (let pivot = 0; pivot < column; pivot += 1) {
        sum -= lower[row][pivot] * lower[column][pivot];
      }

      if (row === column) {
        lower[row][column] = Math.sqrt(Math.max(sum, JITTER));
      } else {
        lower[row][column] = sum / lower[column][column];
      }
    }
  }

  return lower;
}

function solveLower(lower, values) {
  const result = Array(values.length).fill(0);

  for (let row = 0; row < lower.length; row += 1) {
    let sum = values[row];
    for (let column = 0; column < row; column += 1) {
      sum -= lower[row][column] * result[column];
    }
    result[row] = sum / lower[row][row];
  }

  return result;
}

function solveUpperFromLower(lower, values) {
  const result = Array(values.length).fill(0);

  for (let row = lower.length - 1; row >= 0; row -= 1) {
    let sum = values[row];
    for (let column = row + 1; column < lower.length; column += 1) {
      sum -= lower[column][row] * result[column];
    }
    result[row] = sum / lower[row][row];
  }

  return result;
}

function dot(a, b) {
  let total = 0;
  for (let index = 0; index < a.length; index += 1) {
    total += a[index] * b[index];
  }
  return total;
}

function computePosterior() {
  if (state.observations.length === 0) {
    state.posterior = state.grid.map((x) => ({
      x,
      mean: 0,
      sigma: Math.sqrt(SIGNAL_VARIANCE),
      upper: Math.sqrt(SIGNAL_VARIANCE) * EXPLORATION_WEIGHT,
      lower: -Math.sqrt(SIGNAL_VARIANCE) * EXPLORATION_WEIGHT,
    }));
    return;
  }

  const xs = state.observations.map((point) => point.x);
  const ys = state.observations.map((point) => point.y);
  const covariance = xs.map((xA, rowIndex) =>
    xs.map((xB, columnIndex) => {
      const base = kernel(xA, xB);
      if (rowIndex === columnIndex) {
        return base + NOISE_VARIANCE;
      }
      return base;
    }),
  );
  const lower = cholesky(covariance);
  const alpha = solveUpperFromLower(lower, solveLower(lower, ys));

  state.posterior = state.grid.map((x) => {
    const crossCovariance = xs.map((sampleX) => kernel(sampleX, x));
    const mean = dot(crossCovariance, alpha);
    const solved = solveLower(lower, crossCovariance);
    const variance = Math.max(kernel(x, x) - dot(solved, solved), 0);
    const sigma = Math.sqrt(variance);

    return {
      x,
      mean,
      sigma,
      upper: mean + sigma * EXPLORATION_WEIGHT,
      lower: mean - sigma * EXPLORATION_WEIGHT,
    };
  });
}

function selectNextExperiment() {
  if (state.observations.length === 0) {
    return 0.08 + Math.random() * 0.84;
  }

  let bestX = state.grid[0];
  let bestScore = -Infinity;

  state.posterior.forEach((entry) => {
    const nearestDistance = state.observations.reduce((closest, point) => {
      return Math.min(closest, Math.abs(point.x - entry.x));
    }, Infinity);
    if (nearestDistance < 0.018) {
      return;
    }

    const noveltyPenalty = nearestDistance < 0.04 ? 0.35 : 0;
    const score = entry.upper - noveltyPenalty;

    if (score > bestScore) {
      bestScore = score;
      bestX = entry.x;
    }
  });

  if (bestScore === -Infinity) {
    return 0.08 + Math.random() * 0.84;
  }

  return bestX;
}

function updateScale() {
  const values = state.grid.map((x) => state.objective(x));
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const padding = Math.max((maxValue - minValue) * 0.28, 0.35);
  state.yMin = minValue - padding;
  state.yMax = maxValue + padding;
}

function getPosteriorAt(x) {
  if (state.posterior.length === 0) {
    return { mean: 0, sigma: 1 };
  }

  let nearest = state.posterior[0];
  for (const entry of state.posterior) {
    if (Math.abs(entry.x - x) < Math.abs(nearest.x - x)) {
      nearest = entry;
    }
  }
  return nearest;
}

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.floor(window.innerWidth * dpr);
  const height = Math.floor(window.innerHeight * dpr);

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  context.setTransform(1, 0, 0, 1, 0, 0);
  context.scale(dpr, dpr);
}

function toCanvasX(x, bounds) {
  return bounds.left + x * bounds.width;
}

function toCanvasY(y, bounds) {
  const normalized = (y - state.yMin) / (state.yMax - state.yMin);
  return bounds.bottom - normalized * bounds.height;
}

function drawPath(points, bounds, color, width, alpha = 1) {
  context.save();
  context.beginPath();
  points.forEach((point, index) => {
    const canvasX = toCanvasX(point.x, bounds);
    const canvasY = toCanvasY(point.y, bounds);
    if (index === 0) {
      context.moveTo(canvasX, canvasY);
    } else {
      context.lineTo(canvasX, canvasY);
    }
  });
  context.strokeStyle = color;
  context.lineWidth = width;
  context.globalAlpha = alpha;
  context.stroke();
  context.restore();
}

function drawBand(bounds) {
  context.save();
  context.beginPath();
  state.posterior.forEach((point, index) => {
    const canvasX = toCanvasX(point.x, bounds);
    const canvasY = toCanvasY(point.upper, bounds);
    if (index === 0) {
      context.moveTo(canvasX, canvasY);
    } else {
      context.lineTo(canvasX, canvasY);
    }
  });

  for (let index = state.posterior.length - 1; index >= 0; index -= 1) {
    const point = state.posterior[index];
    context.lineTo(toCanvasX(point.x, bounds), toCanvasY(point.lower, bounds));
  }

  context.closePath();
  context.fillStyle = "rgba(116, 240, 201, 0.12)";
  context.fill();
  context.restore();
}

function drawGrid(bounds) {
  context.save();
  context.strokeStyle = "rgba(255, 255, 255, 0.08)";
  context.lineWidth = 1;

  for (let step = 0; step <= 5; step += 1) {
    const x = bounds.left + (bounds.width * step) / 5;
    context.beginPath();
    context.moveTo(x, bounds.top);
    context.lineTo(x, bounds.bottom);
    context.stroke();
  }

  for (let step = 0; step <= 4; step += 1) {
    const y = bounds.top + (bounds.height * step) / 4;
    context.beginPath();
    context.moveTo(bounds.left, y);
    context.lineTo(bounds.right, y);
    context.stroke();
  }

  context.restore();
}

function drawPending(bounds, now) {
  const posterior = getPosteriorAt(state.pendingX);
  const x = toCanvasX(state.pendingX, bounds);
  const y = toCanvasY(posterior.mean, bounds);
  const pulse = 0.65 + 0.35 * Math.sin(now / 240);

  context.save();
  context.strokeStyle = `rgba(255, 214, 102, ${0.4 + pulse * 0.4})`;
  context.lineWidth = 1.5;
  context.setLineDash([8, 8]);
  context.beginPath();
  context.moveTo(x, bounds.top);
  context.lineTo(x, bounds.bottom);
  context.stroke();
  context.setLineDash([]);

  context.beginPath();
  context.arc(x, y, 10 + pulse * 7, 0, Math.PI * 2);
  context.strokeStyle = `rgba(255, 214, 102, ${0.35 + pulse * 0.45})`;
  context.lineWidth = 2.5;
  context.stroke();

  context.beginPath();
  context.arc(x, y, 4, 0, Math.PI * 2);
  context.fillStyle = "#ffd666";
  context.fill();
  context.restore();
}

function drawObservations(bounds) {
  context.save();
  state.observations.forEach((point) => {
    const x = toCanvasX(point.x, bounds);
    const y = toCanvasY(point.y, bounds);

    context.beginPath();
    context.arc(x, y, 6.5, 0, Math.PI * 2);
    context.fillStyle = "#06131f";
    context.fill();

    context.beginPath();
    context.arc(x, y, 4.5, 0, Math.PI * 2);
    context.fillStyle = "#f7fbff";
    context.fill();
  });
  context.restore();
}

function render(now = 0) {
  resizeCanvas();

  const width = window.innerWidth;
  const height = window.innerHeight;
  const bounds = {
    left: Math.min(90, width * 0.08),
    right: width - Math.min(40, width * 0.04),
    top: Math.min(120, height * 0.16),
    bottom: height - Math.min(48, height * 0.08),
  };
  bounds.width = bounds.right - bounds.left;
  bounds.height = bounds.bottom - bounds.top;

  context.clearRect(0, 0, width, height);
  drawGrid(bounds);

  if (state.posterior.length > 0) {
    drawBand(bounds);
  }

  const trueCurve = state.grid.map((x) => ({ x, y: state.objective(x) }));
  const posteriorCurve = state.posterior.map((entry) => ({ x: entry.x, y: entry.mean }));

  drawPath(trueCurve, bounds, "#ffb347", 2, 0.28);
  drawPath(posteriorCurve, bounds, "#74f0c9", 3, 1);
  drawObservations(bounds);
  drawPending(bounds, now);

  window.requestAnimationFrame(render);
}

function runExperiment() {
  const x = state.pendingX;
  const y = state.objective(x);
  state.observations.push({ x, y });
  computePosterior();
  state.pendingX = selectNextExperiment();
}

function resetSimulation() {
  state.grid = createGrid();
  state.objective = createRandomObjective();
  state.observations = [];
  updateScale();
  computePosterior();
  state.pendingX = selectNextExperiment();
}

nextButton.addEventListener("click", runExperiment);
resetButton.addEventListener("click", resetSimulation);

resetSimulation();
const params = new URLSearchParams(window.location.search);
const autoSteps = clamp(Number(params.get("steps")) || 0, 0, 32);
for (let step = 0; step < autoSteps; step += 1) {
  runExperiment();
}
window.requestAnimationFrame(render);
