import base64
import io
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from PIL import Image, ImageOps, ImageChops

app = Flask("Handwritten Digit Recognizer")

# Load the provided sklearn SVM model file (trained on 28x28 MNIST-style flattened images).
MODEL_PATH = "svm_mnist_model.pkl"
clf = joblib.load(MODEL_PATH)

# HTML page served at /
HTML_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Handwritten Digit Recognizer</title>
  <style>
    body { font-family: Arial, Helvetica, sans-serif; display:flex; flex-direction:column; align-items:center; padding:20px; }
    .canvas-wrap { border:1px solid #ccc; display:inline-block; position:relative; }
    canvas { background: white; cursor: crosshair; }
    .controls { margin-top:10px; display:flex; gap:8px; }
    button { padding:8px 12px; font-size:16px; }
    #result { margin-top:12px; font-size:20px; font-weight:600; }
  </style>
</head>
<body>
  <h2>Draw a digit (0-9) and press Predict</h2>
  <div class="canvas-wrap">
    <canvas id="canvas" width="280" height="280"></canvas>
  </div>
  <div class="controls">
    <button id="clearBtn">Clear</button>
    <button id="predictBtn">Predict</button>
    <label> Brush:
      <input id="brushSize" type="range" min="4" max="40" value="18">
    </label>
  </div>
  <div id="result">Prediction: <span id="pred">—</span></div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;
let lastX = 0, lastY = 0;
ctx.lineJoin = ctx.lineCap = 'round';
ctx.lineWidth = 18;
ctx.strokeStyle = 'black';

// initialize white background
ctx.fillStyle = 'white';
ctx.fillRect(0,0,canvas.width,canvas.height);

canvas.addEventListener('pointerdown', (e) => {
  drawing = true;
  const rect = canvas.getBoundingClientRect();
  lastX = e.clientX - rect.left;
  lastY = e.clientY - rect.top;
});

canvas.addEventListener('pointermove', (e) => {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  lastX = x; lastY = y;
});

canvas.addEventListener('pointerup', () => drawing = false);
canvas.addEventListener('pointerleave', () => drawing = false);

document.getElementById('clearBtn').addEventListener('click', () => {
  ctx.fillStyle = 'white';
  ctx.fillRect(0,0,canvas.width,canvas.height);
  document.getElementById('pred').innerText = '—';
});

document.getElementById('brushSize').addEventListener('input', (e) => {
  ctx.lineWidth = Number(e.target.value);
});

document.getElementById('predictBtn').addEventListener('click', async () => {
  const dataURL = canvas.toDataURL('image/png');
  // send to backend
  const resp = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ image: dataURL })
  });
  const j = await resp.json();
  if (j.error) {
    document.getElementById('pred').innerText = 'Error';
    alert(j.error);
  } else {
    document.getElementById('pred').innerText = j.prediction;
  }
});
</script>
</body>
</html>
"""


def preprocess_image_from_bytes(img_bytes):
    """
    Convert raw PNG bytes (from user canvas) into a 1x784 numpy array
    matching MNIST-style 28x28 flattened input the SVM expects.
    Steps:
      - Open image, convert to grayscale
      - Invert (canvas has black strokes on white background -> invert to match MNIST white-on-black)
      - Crop bounding box (tight crop around digit)
      - Resize the digit to fit in a 20x20 box using high-quality LANCZOS (preserves detail)
      - Paste into a 28x28 image centered
      - Shift by center of mass to center the digit (MNIST-style)
      - Normalize pixel values to 0.0-1.0 (float32)
      - Flatten to shape (1, 784)
    """
    img = Image.open(io.BytesIO(img_bytes)).convert('L')  # grayscale
    # Invert colors so strokes are white-on-black (MNIST)
    img = ImageOps.invert(img)

    # Crop to content
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # If bbox is empty (blank canvas), produce blank 28x28
    if img.size[0] == 0 or img.size[1] == 0:
        blank = Image.new('L', (28,28), 0)
        return np.array(blank, dtype=np.float32).reshape(1,-1) / 255.0

    # Resize preserving aspect ratio so the largest dimension becomes 20 pixels
    width, height = img.size
    max_side = max(width, height)
    new_width = int(round(width * 20.0 / max_side))
    new_height = int(round(height * 20.0 / max_side))
    img = img.resize((new_width, new_height), resample=Image.LANCZOS)

    # Put into 28x28 image centered
    new_img = Image.new('L', (28,28), 0)  # black background (because inverted)
    upper_left = ((28 - new_width) // 2, (28 - new_height) // 2)
    new_img.paste(img, upper_left)

    # Shift center of mass to the center (MNIST centering trick)
    arr = np.array(new_img, dtype=np.float32)
    if arr.sum() > 0:
        cy, cx = np.indices(arr.shape)
        total = arr.sum()
        x_center = (cx * arr).sum() / total
        y_center = (cy * arr).sum() / total
        # target center is (14,14)
        shift_x = int(round(14 - x_center))
        shift_y = int(round(14 - y_center))
        new_img = ImageChops.offset(new_img, shift_x, shift_y)

    arr = np.array(new_img, dtype=np.float32)
    arr = arr / 255.0  # normalize to 0.0 - 1.0
    flat = arr.reshape(1, -1)
    return flat


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400
    data_url = data["image"]
    # data_url format: "data:image/png;base64,....."
    if "," not in data_url:
        return jsonify({"error": "Invalid image data"}), 400
    header, b64 = data_url.split(",", 1)
    try:
        img_bytes = base64.b64decode(b64)
    except Exception as e:
        return jsonify({"error": "Could not decode base64 image: " + str(e)}), 400

    # Preprocess to 1x784 vector
    try:
        x = preprocess_image_from_bytes(img_bytes)
    except Exception as e:
        return jsonify({"error": "Preprocessing failed: " + str(e)}), 500

    # Predict using loaded sklearn SVM model
    try:
        pred = clf.predict(x)[0]
        return jsonify({"prediction": int(pred)})
    except Exception as e:
        return jsonify({"error": "Prediction failed: " + str(e)}), 500


if __name__ == "__main__":
    print("Starting app on http://127.0.0.1:5000 — make sure 'svm_mnist_model.pkl' is in this folder.")
    app.run()
