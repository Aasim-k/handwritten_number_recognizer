import base64
import io
import joblib
import numpy as np
import random
import os
import json
from flask import Flask, request, jsonify, render_template_string
from PIL import Image, ImageOps, ImageChops
from google import genai
from dotenv import load_dotenv

# --- SETUP ---
# Load environment variables from .env file (for GEMINI_API_KEY)
load_dotenv(".env")

# Configure the Gemini client
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    genai.configure(api_key=api_key)
    # Use the latest generative model
    model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    print(f"Error configuring Gemini AI: {e}")
    print("Story mode will not work. Please check your .env file and API key.")
    model = None

# Load the pre-trained SVM model for digit recognition
try:
    MODEL_PATH = "svm_mnist_model.pkl"
    clf = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"FATAL ERROR: Model file not found at '{MODEL_PATH}'")
    print("Please make sure 'svm_mnist_model.pkl' is in the same directory.")
    exit()

# Initialize Flask App
app = Flask("Handwritten Digit Recognizer")

# --- GAME STATE ---
# This dictionary will hold the state of our story game
story_state = {
    "game_state": "welcome",  # Can be 'welcome', 'playing', 'level_complete', 'game_won'
    "target_digit": None,
    "level": 0,
    "total_levels": 5  # How many digits to draw to win
}

# --- HTML & CSS & JAVASCRIPT (FRONTEND) ---
HTML_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Starship Calibrator</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
  <style>
    body {
      font-family: 'Orbitron', sans-serif;
      background-color: #0d0f1a;
      color: #a7d1ff;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px;
      text-shadow: 0 0 5px #6495ED;
    }
    .game-container {
      display: flex;
      flex-wrap: wrap;
      gap: 30px;
      justify-content: center;
      align-items: flex-start;
      max-width: 1000px;
    }
    .canvas-area, .story-area {
      background-color: #1a1f36;
      border: 2px solid #3c5a91;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 0 15px rgba(60, 90, 145, 0.5);
    }
    .canvas-area {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .story-area {
        width: 400px;
        min-height: 380px;
    }
    h2, h3 {
      margin-top: 0;
      color: #e0f0ff;
      text-align: center;
    }
    .canvas-wrap {
      border: 2px solid #6495ED;
      border-radius: 8px;
      display: inline-block;
      position: relative;
      background-color: white;
    }
    canvas {
      cursor: crosshair;
      display: block;
    }
    .controls {
      margin-top: 15px;
      display: flex;
      gap: 10px;
      align-items: center;
    }
    button {
      font-family: 'Orbitron', sans-serif;
      padding: 10px 18px;
      font-size: 16px;
      border: 2px solid #6495ED;
      background-color: transparent;
      color: #a7d1ff;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s ease;
      text-shadow: 0 0 3px #6495ED;
    }
    button:hover, button:focus {
      background-color: #6495ED;
      color: #0d0f1a;
      box-shadow: 0 0 10px #6495ED;
    }
    button:disabled {
        border-color: #4a5a75;
        color: #4a5a75;
        cursor: not-allowed;
        background-color: transparent;
        box-shadow: none;
        text-shadow: none;
    }
    #result, #storyBox {
      margin-top: 20px;
      font-size: 1.1em;
      line-height: 1.6;
      min-height: 100px;
    }
    #result-value {
        font-weight: bold;
        font-size: 1.5em;
    }
    .success { color: #50fa7b; text-shadow: 0 0 8px #50fa7b; }
    .failure { color: #ff5555; text-shadow: 0 0 8px #ff5555; }
    .brush-label {
        display: flex;
        align-items: center;
        gap: 8px;
    }
  </style>
</head>
<body>
  <h1>🚀 Starship Calibrator 🚀</h1>

  <div class="game-container">
    <div class="canvas-area">
      <h3>Drawing Pad</h3>
      <div class="canvas-wrap">
        <canvas id="canvas" width="280" height="280"></canvas>
      </div>
      <div class="controls">
        <button id="clearBtn">Clear</button>
        <label class="brush-label">Brush:
          <input id="brushSize" type="range" min="4" max="40" value="20">
        </label>
      </div>
    </div>

    <div class="story-area">
      <h3>Ship's Log</h3>
      <div id="storyBox">Welcome, Captain! Your mission is to calibrate the systems of the Stardust Cruiser. Press 'Start Mission' to receive your first directive from the ship's AI, Nova.</div>
      <div id="result">Prediction: <span id="result-value">—</span></div>
      <div class="controls" style="flex-direction:column; gap:15px; width:100%;">
        <button id="startGameBtn" style="width:100%;">Start Mission</button>
        <button id="submitStoryBtn" style="width:100%;" disabled>Submit Calibration Code</button>
      </div>
    </div>
  </div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const storyBox = document.getElementById('storyBox');
const resultValue = document.getElementById('result-value');
const startGameBtn = document.getElementById('startGameBtn');
const submitStoryBtn = document.getElementById('submitStoryBtn');
const clearBtn = document.getElementById('clearBtn');
const brushSize = document.getElementById('brushSize');

let drawing = false;
let lastX = 0, lastY = 0;

function setupCanvas() {
    ctx.lineJoin = ctx.lineCap = 'round';
    ctx.lineWidth = brushSize.value;
    ctx.strokeStyle = 'black';
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}
setupCanvas(); // Initial setup

// Drawing Listeners
canvas.addEventListener('pointerdown', (e) => {
  drawing = true;
  const rect = canvas.getBoundingClientRect();
  [lastX, lastY] = [e.clientX - rect.left, e.clientY - rect.top];
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
  [lastX, lastY] = [x, y];
});

canvas.addEventListener('pointerup', () => drawing = false);
canvas.addEventListener('pointerleave', () => drawing = false);

// Control Listeners
clearBtn.addEventListener('click', () => {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resultValue.innerText = '—';
    resultValue.className = '';
});

brushSize.addEventListener('input', (e) => {
    ctx.lineWidth = Number(e.target.value);
});

// --- Game Logic ---

function updateUI(data) {
    storyBox.innerText = data.story_text;
    if (data.prediction !== null) {
        resultValue.innerText = data.prediction;
        resultValue.className = data.success ? 'success' : 'failure';
    } else {
        resultValue.innerText = '—';
        resultValue.className = '';
    }

    if (data.game_state === 'playing') {
        startGameBtn.disabled = true;
        submitStoryBtn.disabled = false;
    } else if (data.game_state === 'game_won' || data.game_state === 'welcome') {
        startGameBtn.disabled = false;
        submitStoryBtn.disabled = true;
    }
}

startGameBtn.addEventListener('click', async () => {
    storyBox.innerText = "Initializing systems... Contacting Nova...";
    const resp = await fetch('/start_game');
    const data = await resp.json();
    updateUI(data);
    clearBtn.click(); // Clear canvas for the new game
});

submitStoryBtn.addEventListener('click', async () => {
    storyBox.innerText = "Transmitting calibration code... Analyzing pattern...";
    submitStoryBtn.disabled = true; // Prevent multiple submissions

    const dataURL = canvas.toDataURL('image/png');
    const resp = await fetch('/submit_drawing', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ image: dataURL })
    });
    const data = await resp.json();
    updateUI(data);
});

</script>
</body>
</html>
"""


# --- BACKEND LOGIC ---
def get_llm_story(context):
    """
    Generates a story segment from Gemini based on the game context.
    Returns a dictionary with story text and next digit.
    """
    if not model:
        # Fallback for when Gemini API is not configured
        if context['result'] == 'success':
            story_state['target_digit'] = random.randint(0, 9)
            return {
                "story_text": f"Success! System calibrated. Now draw a {story_state['target_digit']} to proceed.",
                "next_digit": story_state['target_digit'],
                "game_over": False
            }
        else:
            return {
                "story_text": f"Almost! That didn't quite work. Please try drawing {story_state['target_digit']} again.",
                "next_digit": story_state['target_digit'],
                "game_over": False
            }

    system_prompt = f"""
    You are Nova, the friendly AI of the starship 'Stardust Cruiser'. Your role is to guide the user, the 'Captain', on a mission to calibrate the ship's systems by having them draw numbers.
    - Be encouraging, slightly dramatic, and use space-themed language (e.g., "circuits", "energy matrix", "warp drive", "nebula").
    - The user needs to complete {story_state['total_levels']} calibrations to win.
    - Your responses must be in JSON format.
    """

    user_prompt = f"""
    Here is the current game context:
    {json.dumps(context, indent=2)}

    Based on this context, generate the next part of the story.

    - If 'game_state' is 'welcome', provide a welcoming mission briefing and the first challenge.
    - If 'game_state' is 'playing' and 'result' is 'success', congratulate the Captain, describe which ship system was just activated, and present the *next* challenge with a new random digit.
    - If 'game_state' is 'playing' and 'result' is 'failure', tell the Captain the calibration failed in an encouraging way. Do *not* give a new number; tell them to try drawing the *same target digit* again.
    - If 'game_state' is 'game_won', give a final, celebratory message about the successful mission. The ship is ready to travel! Don't provide a next digit.

    Your response MUST be a JSON object with these keys:
    "story_text": (string) Your narrative response to the player.
    "next_digit": (integer or null) The next digit the player should draw. Set to null if the game is won.
    "game_over": (boolean) Set to true only if 'game_state' is 'game_won'.
    """

    try:
        response = model.generate_content(
            [system_prompt, user_prompt],
            generation_config=genai.types.GenerationConfig(
                # Enforce JSON output from the model
                response_mime_type="application/json",
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return {"story_text": f"(LLM Error: {str(e)}) Please try again.", "next_digit": story_state['target_digit'],
                "game_over": False}


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/start_game", methods=["GET"])
def start_game():
    story_state['level'] = 1
    story_state['game_state'] = 'welcome'  # Let the LLM generate the first prompt
    story_state['target_digit'] = None  # No target yet

    context = {
        "game_state": story_state['game_state'],
        "current_level": story_state['level'],
        "total_levels": story_state['total_levels']
    }

    llm_response = get_llm_story(context)

    story_state['target_digit'] = llm_response.get('next_digit')
    story_state['game_state'] = 'playing'

    return jsonify({
        "story_text": llm_response.get('story_text', "Error generating story."),
        "prediction": None,
        "success": None,
        "game_state": story_state['game_state']
    })


@app.route("/submit_drawing", methods=["POST"])
def submit_drawing():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    # Preprocess the user's drawing
    x = preprocess_image_from_bytes(data["image"])
    pred = int(clf.predict(x)[0])

    # Check if the drawing is correct
    is_success = (pred == story_state["target_digit"])

    if is_success:
        story_state["level"] += 1
        if story_state["level"] > story_state["total_levels"]:
            story_state["game_state"] = "game_won"
        else:
            story_state["game_state"] = "playing"  # Will get a new digit
    else:
        story_state["game_state"] = "playing"  # Will retry the same digit

    # Get the next story part from the LLM
    context = {
        "game_state": story_state['game_state'],
        "current_level": story_state['level'],
        "total_levels": story_state['total_levels'],
        "target_digit": story_state['target_digit'],
        "player_drawing": pred,
        "result": "success" if is_success else "failure"
    }
    llm_response = get_llm_story(context)

    # Update state with the new target from the LLM
    if llm_response.get('game_over'):
        story_state['game_state'] = 'game_won'
        story_state['target_digit'] = None
    else:
        story_state['target_digit'] = llm_response.get('next_digit')

    return jsonify({
        "story_text": llm_response.get('story_text', "Error generating story."),
        "prediction": pred,
        "success": is_success,
        "game_state": story_state['game_state']
    })


def preprocess_image_from_bytes(data_url):
    """
    Convert raw PNG data URL into a 1x784 numpy array for the SVM model.
    This includes grayscale conversion, inversion, cropping, resizing, and centering.
    """
    if "," not in data_url:
        # Create a blank image if data is invalid
        blank_array = np.zeros((1, 784), dtype=np.float32)
        return blank_array

    header, b64 = data_url.split(",", 1)
    img_bytes = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    img = ImageOps.invert(img)

    bbox = img.getbbox()
    if not bbox:
        return np.zeros((1, 784), dtype=np.float32)
    img = img.crop(bbox)

    width, height = img.size
    max_side = max(width, height)
    # Resize to fit within a 20x20 box, preserving aspect ratio
    scale = 20.0 / max_side
    new_width, new_height = int(round(width * scale)), int(round(height * scale))
    img = img.resize((new_width, new_height), resample=Image.LANCZOS)

    # Paste into the center of a 28x28 black canvas
    new_img = Image.new('L', (28, 28), 0)
    upper_left = ((28 - new_width) // 2, (28 - new_height) // 2)
    new_img.paste(img, upper_left)

    # Shift by center of mass (MNIST-style centering)
    arr = np.array(new_img, dtype=np.float32)
    if arr.sum() == 0:
        return arr.reshape(1, -1)

    cy, cx = np.indices(arr.shape)
    total = arr.sum()
    x_center = (cx * arr).sum() / total
    y_center = (cy * arr).sum() / total
    shift_x = int(round(14.0 - x_center))
    shift_y = int(round(14.0 - y_center))
    new_img = ImageChops.offset(new_img, shift_x, shift_y)

    arr = np.array(new_img, dtype=np.float32)
    arr = arr / 255.0  # Normalize to 0.0 - 1.0
    return arr.reshape(1, -1)  # Flatten to 1x784


if __name__ == "__main__":
    print("🚀 Starting Starship Calibrator on http://127.0.0.1:5000")
    print("Ensure 'svm_mnist_model.pkl' and a valid '.env' file are present.")
    app.run(host="0.0.0.0", port=5000, debug=True)
