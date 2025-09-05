import base64
import io
import joblib
import numpy as np
import random
import os
import json
import time
from flask import Flask, request, jsonify, render_template_string
from PIL import Image, ImageOps, ImageChops
from google.generativeai import GenerativeModel, configure
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv

# --- SETUP ---
load_dotenv(".env")

# Configure the Gemini client
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    configure(api_key=api_key)
    model = GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error configuring Gemini AI: {e}")
    model = None

# Load the pre-trained SVM model
try:
    MODEL_PATH = "svm_mnist_model.pkl"
    clf = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"FATAL ERROR: Model file not found at '{MODEL_PATH}'")
    exit()

app = Flask("AI Containment Game")


# --- DEFAULT GAME STATE ---
def get_default_state():
    return {
        "game_state": "welcome",
        "target_digit": None,
        "level": 0,
        "total_levels": 5,
        "time_limit": 90,
        "start_time": None,
        "attempts_remaining": 3,
        "max_attempts": 3,
        "game_sequence": []  # The pre-generated sequence of correct digits
    }


story_state = get_default_state()

# --- FRONTEND (HTML, CSS, JS) ---
HTML_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>AI Containment Protocol</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --main-bg: #0d1a26;
      --container-bg: #0a141e;
      --border-color: #ff4d4d;
      --text-color: #e6e6e6;
      --glow-color: rgba(255, 77, 77, 0.4);
      --success-color: #00ff41;
      --success-glow: rgba(0, 255, 65, 0.4);
      --warning-color: #ffaa00;
      --warning-glow: rgba(255, 170, 0, 0.5);
    }
    body {
      font-family: 'Fira Code', monospace;
      background-color: var(--main-bg);
      color: var(--text-color);
      display: flex; flex-direction: column; align-items: center; padding: 20px;
      text-shadow: 0 0 3px rgba(255, 255, 255, 0.1);
      background-image: radial-gradient(circle at 15% 85%, var(--glow-color) 0%, transparent 40%);
      transition: background-image 1s ease-in-out;
    }
    .game-container {
      display: flex; flex-wrap: wrap; gap: 30px; justify-content: center; align-items: flex-start; max-width: 1000px;
    }
    .canvas-area, .story-area {
      background-color: var(--container-bg); border: 1px solid var(--border-color);
      padding: 20px; box-shadow: 0 0 20px var(--glow-color); border-radius: 8px;
      transition: border-color 1s, box-shadow 1s;
    }
    .canvas-area { display: flex; flex-direction: column; align-items: center; min-width: 320px; }
    .story-area { width: 400px; min-height: 450px; }
    h1, h2, h3 { margin-top: 0; text-align: center; text-transform: uppercase; letter-spacing: 2px; color: var(--border-color); transition: color 1s, text-shadow 1s; }
    h1 { font-size: 2.2em; margin-bottom: 30px; text-shadow: 0 0 15px var(--glow-color); }
    .canvas-wrap { border: 2px solid var(--border-color); background-color: #f0f0f0; border-radius: 4px; }
    canvas { cursor: crosshair; display: block; }
    .controls { margin-top: 15px; display: flex; gap: 15px; align-items: center; justify-content: center; width: 100%;}
    button {
      font-family: 'Fira Code', monospace; padding: 12px 20px; font-size: 16px;
      border: 2px solid var(--border-color); background-color: transparent; color: var(--border-color);
      cursor: pointer; transition: all 0.3s ease; text-shadow: 0 0 5px var(--glow-color);
      border-radius: 4px; text-transform: uppercase; letter-spacing: 1px;
    }
    button:hover:not(:disabled) {
      background-color: var(--border-color); color: var(--main-bg); box-shadow: 0 0 15px var(--glow-color);
      transform: translateY(-2px);
    }
    button:disabled { border-color: #553333; color: #885555; cursor: not-allowed; text-shadow: none; transform: none; }
    #resetBtn { border-color: var(--warning-color); color: var(--warning-color); }
    #resetBtn:hover:not(:disabled) { background-color: var(--warning-color); color: var(--main-bg); box-shadow: 0 0 15px var(--warning-glow); }
    #storyBox { 
      font-size: 1em; line-height: 1.6; min-height: 150px; background-color: rgba(255, 77, 77, 0.05);
      padding: 15px; border-radius: 4px; border: 1px solid rgba(255, 77, 77, 0.2);
    }
    #timer { font-size: 2.5em; font-weight: bold; margin: 15px 0; text-align: center; }
    .game-info { display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 1.1em; font-weight: bold; }
    .attempts-info { color: var(--warning-color); }
    .attempts-critical { color: var(--failure-color); animation: pulse 1s infinite; }
    .success { color: var(--success-color); text-shadow: 0 0 8px var(--success-glow); }
    .failure { color: var(--border-color); text-shadow: 0 0 8px var(--glow-color); }
    .low-time { color: var(--border-color); animation: pulse 1s infinite; }
    @keyframes pulse { 50% { text-shadow: 0 0 20px var(--glow-color); color: #fff; } }
    #timer-bar-container {
        width: 100%; height: 10px; background-color: rgba(255, 77, 77, 0.2);
        border: 1px solid var(--border-color); border-radius: 5px; margin-top: 10px;
    }
    #timer-bar {
        width: 100%; height: 100%; background-color: var(--success-color);
        border-radius: 4px; transition: width 1s linear, background-color 1s linear;
    }
    body.contained {
        background-image: radial-gradient(circle at 15% 85%, var(--success-glow) 0%, transparent 40%);
    }
    body.contained .story-area,
    body.contained .canvas-area {
        border-color: var(--success-color); box-shadow: 0 0 25px var(--success-glow);
    }
    body.contained h1,
    body.contained h3 {
        color: var(--success-color); text-shadow: 0 0 15px var(--success-glow);
    }
  </style>
</head>
<body>
  <h1>// AI CONTAINMENT PROTOCOL //</h1>
  <div class="game-container">
    <div class="canvas-area">
      <h3>Firewall Input</h3>
      <div class="canvas-wrap"><canvas id="canvas" width="280" height="280"></canvas></div>
      <div class="controls"><button id="clearBtn">Clear Pad</button></div>
    </div>
    <div class="story-area">
      <h3>> Director Thorne's Channel</h3>
      <div class="game-info">
        <span id="levelDisplay">Layer: -/5</span>
        <span id="attemptsDisplay" class="attempts-info">Stability: -/3</span>
      </div>
      <div id="storyBox">URGENT: Rogue AI "SYNAPSE" is attempting a containment breach. Your mission is to reinforce the quarantine firewalls by entering a series of counter-protocol codes. The system's integrity is failing. Press 'Deploy' to begin.</div>
      <div id="timer">Time until breach: --</div>
      <div id="timer-bar-container"><div id="timer-bar"></div></div>
      <div id="result">Input Analysis: <span id="result-value">—</span></div>
      <div class="controls" style="flex-direction:column; gap:15px;">
        <button id="startGameBtn" style="width:100%;">Deploy</button>
        <button id="submitStoryBtn" style="width:100%;" disabled>Transmit Code</button>
        <button id="resetBtn" style="width:100%;">Reset Mission</button>
      </div>
    </div>
  </div>
<script>
const canvas = document.getElementById('canvas'), ctx = canvas.getContext('2d');
const storyBox = document.getElementById('storyBox'), resultValue = document.getElementById('result-value');
const timerDisplay = document.getElementById('timer'), levelDisplay = document.getElementById('levelDisplay'), attemptsDisplay = document.getElementById('attemptsDisplay');
const startGameBtn = document.getElementById('startGameBtn'), submitStoryBtn = document.getElementById('submitStoryBtn'), resetBtn = document.getElementById('resetBtn');
const clearBtn = document.getElementById('clearBtn');
let drawing = false, lastX = 0, lastY = 0, timerInterval = null;

function setupCanvas() {
    ctx.lineJoin = ctx.lineCap = 'round';
    ctx.lineWidth = 20;
    ctx.strokeStyle = 'black'; ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}
setupCanvas();

canvas.addEventListener('pointerdown', (e) => { drawing = true; const rect = canvas.getBoundingClientRect(); [lastX, lastY] = [e.clientX - rect.left, e.clientY - rect.top]; });
canvas.addEventListener('pointermove', (e) => { if (!drawing) return; const rect = canvas.getBoundingClientRect(); const x = e.clientX - rect.left, y = e.clientY - rect.top; ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(x, y); ctx.stroke(); [lastX, lastY] = [x, y]; });
canvas.addEventListener('pointerup', () => drawing = false);
canvas.addEventListener('pointerleave', () => drawing = false);
clearBtn.addEventListener('click', () => { ctx.fillRect(0, 0, canvas.width, canvas.height); resultValue.innerText = '—'; resultValue.className = ''; });

function stopTimer() { if (timerInterval) clearInterval(timerInterval); timerInterval = null; timerDisplay.classList.remove('low-time'); }

function updateUI(data) {
    storyBox.innerText = data.story_text; // Set text for all states initially
    resultValue.innerText = data.prediction !== null ? data.prediction : '—';
    resultValue.className = data.success ? 'success' : (data.success === false ? 'failure' : '');
    levelDisplay.innerText = `Layer: ${data.level || '-'}/${data.total_levels || 5}`;
    attemptsDisplay.innerText = `Stability: ${data.attempts_remaining || '-'}/${data.max_attempts || 3}`;
    attemptsDisplay.className = data.attempts_remaining <= 1 ? 'attempts-critical' : 'attempts-info';
    
    document.body.classList.remove('contained');

    const isGameOver = data.game_state === 'game_won' || data.game_state === 'time_up' || data.game_state === 'welcome';
    if (isGameOver) {
        stopTimer();
        startGameBtn.disabled = false;
        submitStoryBtn.disabled = true;
        if (data.game_state === 'time_up') {
            timerDisplay.innerText = "BREACH DETECTED - MISSION FAILED";
        }
        if (data.game_state === 'game_won') {
            timerDisplay.innerText = "SYNAPSE CONTAINED!";
            document.body.classList.add('contained');
            typeWriter(storyBox, data.story_text, 50); // Animate final message
        }
    } else { // Playing
        startGameBtn.disabled = true;
        submitStoryBtn.disabled = false;
    }
}

startGameBtn.addEventListener('click', async () => {
    storyBox.innerText = "Initializing quarantine protocols...";
    const resp = await fetch('/start_game');
    const data = await resp.json();
    updateUI(data);
    clearBtn.click();

    let timeLeft = data.time_limit;
    const totalTime = data.time_limit;
    const timerBar = document.getElementById('timer-bar');
    timerBar.style.backgroundColor = 'var(--success-color)'; // Reset bar color
    timerBar.style.width = '100%';

    timerDisplay.innerText = `Time until breach: ${timeLeft}s`;
    stopTimer();
    timerInterval = setInterval(() => {
        timeLeft--;
        timerDisplay.innerText = `Time until breach: ${timeLeft}s`;
        const percentageLeft = (timeLeft / totalTime) * 100;
        timerBar.style.width = `${percentageLeft}%`;

        if (timeLeft <= totalTime * 0.25) {
            timerBar.style.backgroundColor = 'var(--border-color)';
            timerDisplay.classList.add('low-time');
        } else if (timeLeft <= totalTime * 0.5) {
            timerBar.style.backgroundColor = 'var(--warning-color)';
        } else {
            timerBar.style.backgroundColor = 'var(--success-color)';
        }

        if (timeLeft <= 0) {
            stopTimer();
            timerDisplay.innerText = "BREACH DETECTED - MISSION FAILED";
            submitStoryBtn.disabled = true;
        }
    }, 1000);
});

submitStoryBtn.addEventListener('click', async () => {
    storyBox.innerText = "Analyzing neural input... Transmitting counter-protocol...";
    submitStoryBtn.disabled = true;
    const dataURL = canvas.toDataURL('image/png');
    const resp = await fetch('/submit_drawing', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ image: dataURL })
    });
    const data = await resp.json();
    clearBtn.click();
    updateUI(data);
    if (data.game_state === 'playing') submitStoryBtn.disabled = false;
});

resetBtn.addEventListener('click', async () => {
    const resp = await fetch('/reset_game');
    const data = await resp.json();
    timerDisplay.innerText = "Time until breach: --";
    document.getElementById('timer-bar').style.width = '100%';
    document.getElementById('timer-bar').style.backgroundColor = 'var(--success-color)';
    updateUI(data);
});

function typeWriter(element, text, speed) {
    let i = 0;
    element.innerHTML = "";
    function type() {
        if (i < text.length) {
            element.innerHTML += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }
    type();
}
</script>
</body>
</html>
"""


# --- BACKEND LOGIC ---
def get_llm_story(context):
    if not model:  # Fallback if Gemini is offline
        if context['game_state'] == 'welcome':
            return {"story_text": f"Welcome, Specialist. Your first counter-protocol is {context['next_digit']}."}
        elif context['result'] == 'success':
            return {"story_text": f"Success! The next firewall code is {context['next_digit']}."}
        else:
            return {"story_text": f"Failure! Firewall integrity dropping. Retry code {context['next_digit']}."}

    system_prompt = """
    You are a story generator for a tense AI containment game. Your response MUST be a single JSON object.
    There are two characters:
    1. Director Thorne: A calm, professional mission handler. All primary text should be from his perspective.
    2. SYNAPSE: A rogue AI. It can occasionally interject with a taunt, especially on failure.

    Your task is to provide narrative text based on the game's context.
    """
    user_prompt = f"""
    Here is the current mission status:
    {json.dumps(context, indent=2)}

    Based on the context, write the "story_text".
    - WELCOME: Brief the player on their mission to contain SYNAPSE and give them their first counter-protocol code, which is the provided 'next_digit'.
    - SUCCESS: Inform the player they've reinforced a firewall layer. If the game is won, deliver a victory message. If continuing, tell them the next required code is the provided 'next_digit'.
    - FAILURE: State that the code was rejected and firewall stability is dropping. Tell them to re-enter the SAME code, which is the provided 'next_digit'. You can optionally include a short, taunting message from SYNAPSE in quotes.
    - TIME_UP: Deliver a mission failed message.

    CRITICAL: Your entire response must be ONLY a valid JSON object like this: {{"story_text": "Your narrative here."}}
    """
    try:
        response = model.generate_content(
            [system_prompt, user_prompt],
            generation_config=GenerationConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"LLM Error: {str(e)}, using fallback")
        return get_llm_story(context)  # Call self for fallback


def process_submission(predicted_digit):
    is_success = (predicted_digit == story_state["target_digit"])
    context = {"player_input": predicted_digit}

    if is_success:
        story_state["level"] += 1
        story_state["attempts_remaining"] = story_state["max_attempts"]
        if story_state["level"] > story_state["total_levels"]:
            story_state["game_state"] = "game_won"
            context["next_digit"] = None
        else:
            story_state["game_state"] = "playing"
            next_index = story_state["level"] - 1
            story_state["target_digit"] = story_state["game_sequence"][next_index]
            context["next_digit"] = story_state["target_digit"]
        context["result"] = "success"
    else:
        story_state["attempts_remaining"] -= 1
        if story_state["attempts_remaining"] <= 0:
            story_state["game_state"] = "time_up"  # Game over due to instability
            context["next_digit"] = None
        else:
            story_state["game_state"] = "playing"
            context["next_digit"] = story_state["target_digit"]
        context["result"] = "failure"

    context.update({
        "game_state": story_state['game_state'],
        "level": story_state['level'],
        "attempts_remaining": story_state['attempts_remaining']
    })

    llm_response = get_llm_story(context)

    return jsonify({
        "story_text": llm_response.get('story_text'),
        "prediction": predicted_digit,
        "success": is_success,
        "game_state": story_state['game_state'],
        "level": story_state['level'],
        "total_levels": story_state['total_levels'],
        "attempts_remaining": story_state['attempts_remaining'],
        "max_attempts": story_state['max_attempts']
    })


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/start_game", methods=["GET"])
def start_game():
    story_state.update(get_default_state())
    game_sequence = [random.randint(0, 9) for _ in range(story_state['total_levels'])]
    story_state.update({
        "level": 1, "game_state": 'playing',
        "start_time": time.time(), "game_sequence": game_sequence,
        "target_digit": game_sequence[0]
    })

    context = {
        "game_state": "welcome",
        "next_digit": story_state['target_digit']
    }
    llm_response = get_llm_story(context)

    return jsonify({**story_state, **llm_response})


@app.route("/reset_game", methods=["GET"])
def reset_game():
    global story_state
    story_state = get_default_state()
    return jsonify({
        **story_state,
        "story_text": "URGENT: Rogue AI \"SYNAPSE\" is attempting a containment breach. Your mission is to reinforce the quarantine firewalls by entering a series of counter-protocol codes. The system's integrity is failing. Press 'Deploy' to begin."
    })


@app.route("/submit_drawing", methods=["POST"])
def submit_drawing():
    time_elapsed = time.time() - story_state.get('start_time', 0)
    if time_elapsed >= story_state['time_limit']:
        story_state['game_state'] = 'time_up'
        llm_response = get_llm_story({"game_state": "time_up"})
        return jsonify({**story_state, **llm_response, "success": False})

    data = request.get_json()
    x = preprocess_image_from_bytes(data["image"])
    pred = int(clf.predict(x)[0])
    return process_submission(pred)


def preprocess_image_from_bytes(data_url):
    try:
        if "," not in data_url: return np.zeros((1, 784), dtype=np.float32)
        _, b64 = data_url.split(",", 1)
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        img = ImageOps.invert(img)
        bbox = img.getbbox()
        if not bbox: return np.zeros((1, 784), dtype=np.float32)
        img = img.crop(bbox)
        scale = 20.0 / max(img.size)
        new_size = (int(round(img.size[0] * scale)), int(round(img.size[1] * scale)))
        img = img.resize(new_size, resample=Image.LANCZOS)
        new_img = Image.new('L', (28, 28), 0)
        new_img.paste(img, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))
        arr = np.array(new_img, dtype=np.float32)
        if arr.sum() == 0: return arr.reshape(1, -1)
        cy, cx = np.indices(arr.shape)
        x_center = (cx * arr).sum() / arr.sum()
        y_center = (cy * arr).sum() / arr.sum()
        shift_x = int(round(14.0 - x_center))
        shift_y = int(round(14.0 - y_center))
        new_img = ImageChops.offset(new_img, shift_x, shift_y)
        final_arr = np.array(new_img, dtype=np.float32)
        return (final_arr.reshape(1, -1) / 255.0)
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return np.zeros((1, 784), dtype=np.float32)


if __name__ == "__main__":
    print("=" * 60)
    print(">> AI CONTAINMENT PROTOCOL - SYSTEM ONLINE <<")
    print(f"Server running at: http://127.0.0.1:5000")
    print(f"SVM Model Status: {'LOADED' if clf else 'MISSING'}")
    print(f"LLM Connection: {'ACTIVE' if model else 'OFFLINE'}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)