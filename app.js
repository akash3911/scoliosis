const apiUrlInput = document.getElementById("apiUrl");
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const resultCanvas = document.getElementById("resultCanvas");
const resultMessage = document.getElementById("resultMessage");
const anglesPanel = document.getElementById("anglesPanel");
const submitBtn = document.getElementById("submitBtn");
const resultContext = resultCanvas.getContext("2d");

const imageState = {
  fileUrl: null,
  objectUrl: null,
};

function setResultMessage(message) {
  resultMessage.textContent = message;
}

function renderAngles(angles, curveType) {
  const entries = angles && typeof angles === "object" ? Object.entries(angles) : [];

  if (!entries.length) {
    anglesPanel.innerHTML = "";
    return;
  }

  const curveRow = curveType
    ? `<div class="angle-row"><span class="angle-label">Curve</span><span class="angle-value">${curveType}</span></div>`
    : "";

  const angleRows = entries
    .map(([key, value]) => {
      const label = key.toUpperCase();
      const angle = Number(value?.angle);
      const displayValue = Number.isFinite(angle) ? `${angle.toFixed(2)}°` : "N/A";
      const idxs = Array.isArray(value?.idxs) ? value.idxs.join(", ") : "";
      const meta = idxs ? ` <span class="angle-meta">(${idxs})</span>` : "";
      return `<div class="angle-row"><span class="angle-label">${label}${meta}</span><span class="angle-value">${displayValue}</span></div>`;
    })
    .join("");

  anglesPanel.innerHTML = curveRow + angleRows;
}

function clearCanvas() {
  resultContext.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
  resultCanvas.width = 0;
  resultCanvas.height = 0;
}

function drawLine(x1, y1, x2, y2, color = "#0f766e", width = 3) {
  resultContext.strokeStyle = color;
  resultContext.lineWidth = width;
  resultContext.beginPath();
  resultContext.moveTo(x1, y1);
  resultContext.lineTo(x2, y2);
  resultContext.stroke();
}

function drawPoint(x, y, color = "#dc2626", radius = 4) {
  resultContext.fillStyle = color;
  resultContext.beginPath();
  resultContext.arc(x, y, radius, 0, Math.PI * 2);
  resultContext.fill();
}

function drawLabel(text, x, y) {
  resultContext.font = "600 14px 'Segoe UI', sans-serif";
  resultContext.fillStyle = "#111827";
  resultContext.fillText(text, x + 8, y - 8);
}

function drawPrediction(image, data) {
  resultCanvas.width = image.naturalWidth;
  resultCanvas.height = image.naturalHeight;

  resultContext.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
  resultContext.drawImage(image, 0, 0, resultCanvas.width, resultCanvas.height);

  const detections = Array.isArray(data?.detections) ? data.detections : [];
  const landmarks = Array.isArray(data?.landmarks) ? data.landmarks : [];
  const midpointLines = Array.isArray(data?.midpoint_lines) ? data.midpoint_lines : [];

  midpointLines.forEach((line) => {
    if (!Array.isArray(line) || line.length < 2) {
      return;
    }

    const [start, end] = line;
    if (!Array.isArray(start) || !Array.isArray(end)) {
      return;
    }

    drawLine(start[0], start[1], end[0], end[1], "#0f766e", 3);
  });

  detections.forEach((det) => {
    const x = Number(det.xmin);
    const y = Number(det.ymin);
    const width = Number(det.xmax) - x;
    const height = Number(det.ymax) - y;

    resultContext.strokeStyle = "#16a34a";
    resultContext.lineWidth = 3;
    resultContext.strokeRect(x, y, width, height);
    drawLabel(`${det.name ?? "vert"} ${(Number(det.confidence) || 0).toFixed(2)}`, x, y);
  });

  for (let index = 0; index < landmarks.length; index += 2) {
    const x = Number(landmarks[index]);
    const y = Number(landmarks[index + 1]);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      drawPoint(x, y);
    }
  }
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files && imageInput.files[0];
  if (!file) {
    preview.style.display = "none";
    imageState.fileUrl = null;
    anglesPanel.innerHTML = "";
    return;
  }

  if (imageState.objectUrl) {
    URL.revokeObjectURL(imageState.objectUrl);
  }

  imageState.objectUrl = URL.createObjectURL(file);
  imageState.fileUrl = imageState.objectUrl;
  preview.src = imageState.objectUrl;
  preview.style.display = "block";
  clearCanvas();
  setResultMessage("Ready to plot prediction.");
});

submitBtn.addEventListener("click", async () => {
  const file = imageInput.files && imageInput.files[0];
  if (!file) {
    setResultMessage("Please choose an image first.");
    return;
  }

  const url = apiUrlInput.value.trim();
  if (!url) {
    setResultMessage("Please provide a valid API URL.");
    return;
  }

  submitBtn.disabled = true;
  setResultMessage("Running prediction...");

  try {
    const formData = new FormData();
    formData.append("image", file);

    const response = await fetch(url, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const data = await response.json();
    renderAngles(data.angles, data.curve_type);

    const image = new Image();
    image.onload = () => {
      drawPrediction(image, data);
      setResultMessage("Prediction plotted on the image with angles shown below.");
    };
    image.onerror = () => {
      setResultMessage("Prediction received, but the image could not be rendered.");
    };

    image.src = imageState.fileUrl || URL.createObjectURL(file);
  } catch (err) {
    setResultMessage(`Error: ${err instanceof Error ? err.message : "Unknown error"}`);
  } finally {
    submitBtn.disabled = false;
  }
});
