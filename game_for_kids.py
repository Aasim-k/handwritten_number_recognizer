import base64
import io
import joblib
import numpy as np
import json
import random
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template_string, session
from PIL import Image, ImageOps, ImageChops
import os
from datetime import datetime
from dotenv import load_dotenv


load_dotenv(".env")
app = Flask("Number Learning Adventure")
app.secret_key = "your-secret-key-change-this"

# Configure Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Load the SVM model
MODEL_PATH = "svm_mnist_model.pkl"
clf = joblib.load(MODEL_PATH)

# Game configuration
LEVELS = {
    "beginner": {"range": [0, 5], "challenges": 3, "time_limit": 60},
    "intermediate": {"range": [0, 9], "challenges": 5, "time_limit": 45},
    "advanced": {"range": [0, 9], "challenges": 7, "time_limit": 30}
}

# HTML page with game interface
HTML_PAGE = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Number Learning Adventure 🚀</title>
    <style>
        body { 
            font-family: 'Comic Sans MS', cursive, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0; padding: 20px; color: white; min-height: 100vh;
        }
        .container { max-width: 800px; margin: 0 auto; text-align: center; }
        .header { margin-bottom: 30px; }
        .game-area { 
            background: rgba(255,255,255,0.95); 
            border-radius: 20px; 
            padding: 30px; 
            color: #333;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .canvas-wrap { 
            border: 3px solid #4CAF50; 
            border-radius: 15px;
            display: inline-block; 
            margin: 20px 0;
            background: white;
        }
        canvas { 
            background: white; 
            cursor: crosshair; 
            border-radius: 12px;
        }
        .controls { 
            margin: 20px 0; 
            display: flex; 
            gap: 15px; 
            justify-content: center;
            flex-wrap: wrap;
        }
        button { 
            padding: 12px 20px; 
            font-size: 16px; 
            border: none; 
            border-radius: 25px; 
            cursor: pointer; 
            font-weight: bold;
            transition: all 0.3s;
        }
        .btn-primary { background: #4CAF50; color: white; }
        .btn-secondary { background: #2196F3; color: white; }
        .btn-danger { background: #f44336; color: white; }
        button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.3); }
        .game-info { 
            display: flex; 
            justify-content: space-between; 
            margin: 20px 0;
            flex-wrap: wrap;
            gap: 15px;
        }
        .info-box { 
            background: #f0f0f0; 
            padding: 15px; 
            border-radius: 10px; 
            flex: 1; 
            min-width: 120px;
        }
        .challenge-box { 
            background: #e3f2fd; 
            padding: 20px; 
            border-radius: 15px; 
            margin: 20px 0;
            border-left: 5px solid #2196F3;
        }
        .result-box { 
            font-size: 24px; 
            font-weight: bold; 
            margin: 20px 0; 
            padding: 15px;
            border-radius: 10px;
        }
        .correct { background: #c8e6c9; color: #2e7d32; }
        .incorrect { background: #ffcdd2; color: #c62828; }
        .level-selector { margin: 20px 0; }
        .level-btn { margin: 0 10px; }
        .progress-bar { 
            background: #ddd; 
            height: 20px; 
            border-radius: 10px; 
            overflow: hidden; 
            margin: 10px 0;
        }
        .progress-fill { 
            height: 100%; 
            background: #4CAF50; 
            transition: width 0.3s; 
        }
        .achievements { 
            display: flex; 
            gap: 10px; 
            justify-content: center; 
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .achievement { 
            padding: 8px 15px; 
            background: gold; 
            color: #333; 
            border-radius: 20px; 
            font-size: 14px;
            font-weight: bold;
        }
        .hidden { display: none; }
        .story-box { 
            background: #fff3e0; 
            padding: 20px; 
            border-radius: 15px; 
            margin: 20px 0;
            border-left: 5px solid #ff9800;
            text-align: left;
        }
        .timer { 
            font-size: 20px; 
            font-weight: bold; 
            color: #f44336; 
        }
        @media (max-width: 600px) {
            .game-info { flex-direction: column; }
            .controls { flex-direction: column; align-items: center; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Number Learning Adventure 🚀</h1>
            <p>Draw numbers and complete fun challenges!</p>
        </div>

        <div class="game-area">
            <!-- Level Selection -->
            <div id="levelSelection" class="level-selector">
                <h2>Choose Your Adventure Level!</h2>
                <button class="btn-primary level-btn" onclick="startGame('beginner')">🌟 Beginner (0-5)</button>
                <button class="btn-secondary level-btn" onclick="startGame('intermediate')">⭐ Intermediate (0-9)</button>
                <button class="btn-danger level-btn" onclick="startGame('advanced')">🔥 Advanced (Quick!)</button>
            </div>

            <!-- Game Interface -->
            <div id="gameInterface" class="hidden">
                <div class="game-info">
                    <div class="info-box">
                        <div>Level: <span id="currentLevel">-</span></div>
                    </div>
                    <div class="info-box">
                        <div>Challenge: <span id="currentChallenge">0</span>/<span id="totalChallenges">0</span></div>
                    </div>
                    <div class="info-box">
                        <div>Score: <span id="score">0</span></div>
                    </div>
                    <div class="info-box">
                        <div class="timer">Time: <span id="timeLeft">--</span>s</div>
                    </div>
                </div>

                <div class="progress-bar">
                    <div id="progressFill" class="progress-fill" style="width: 0%"></div>
                </div>

                <div id="storyBox" class="story-box">
                    <h3>📖 Story</h3>
                    <p id="storyText">Loading your adventure...</p>
                </div>

                <div id="challengeBox" class="challenge-box">
                    <h3>🎯 Your Mission</h3>
                    <p id="challengeText">Loading challenge...</p>
                </div>

                <div class="canvas-wrap">
                    <canvas id="canvas" width="280" height="280"></canvas>
                </div>

                <div class="controls">
                    <button id="clearBtn" class="btn-secondary">🗑️ Clear</button>
                    <button id="submitBtn" class="btn-primary">✨ Submit Answer</button>
                    <button id="hintBtn" class="btn-secondary">💡 Hint</button>
                    <label>
                        Brush Size:
                        <input id="brushSize" type="range" min="8" max="30" value="18">
                    </label>
                </div>

                <div id="resultBox" class="result-box hidden">
                    <div id="resultText">Result</div>
                </div>

                <div class="achievements">
                    <div id="achievements"></div>
                </div>

                <div class="controls">
                    <button id="nextBtn" class="btn-primary hidden">➡️ Next Challenge</button>
                    <button id="newGameBtn" class="btn-secondary">🔄 New Game</button>
                </div>
            </div>

            <!-- Game Complete -->
            <div id="gameComplete" class="hidden">
                <h2>🎉 Adventure Complete!</h2>
                <div id="finalScore"></div>
                <div id="finalAchievements"></div>
                <button class="btn-primary" onclick="showLevelSelection()">🎮 Play Again</button>
            </div>
        </div>
    </div>

<script>
let gameState = {
    level: '',
    currentChallenge: 0,
    totalChallenges: 0,
    score: 0,
    timeLeft: 0,
    timer: null,
    challenges: [],
    currentAnswer: null,
    achievements: new Set()
};

// Canvas setup
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;
let lastX = 0, lastY = 0;

function initCanvas() {
    ctx.lineJoin = ctx.lineCap = 'round';
    ctx.lineWidth = 18;
    ctx.strokeStyle = 'black';
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Canvas event listeners
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
    lastX = x; 
    lastY = y;
});

canvas.addEventListener('pointerup', () => drawing = false);
canvas.addEventListener('pointerleave', () => drawing = false);

// Game functions
async function startGame(level) {
    gameState.level = level;
    gameState.currentChallenge = 0;
    gameState.score = 0;
    gameState.achievements.clear();

    document.getElementById('levelSelection').classList.add('hidden');
    document.getElementById('gameInterface').classList.remove('hidden');
    document.getElementById('gameComplete').classList.add('hidden');

    // Initialize game
    await initializeGame();
    startTimer();
    initCanvas();
}

async function initializeGame() {
    try {
        const response = await fetch('/start_game', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ level: gameState.level })
        });
        const data = await response.json();

        gameState.challenges = data.challenges;
        gameState.totalChallenges = data.total_challenges;
        gameState.timeLeft = data.time_limit;

        updateUI();
        loadCurrentChallenge();
    } catch (error) {
        console.error('Failed to initialize game:', error);
    }
}

function loadCurrentChallenge() {
    if (gameState.currentChallenge >= gameState.totalChallenges) {
        completeGame();
        return;
    }

    const challenge = gameState.challenges[gameState.currentChallenge];
    document.getElementById('storyText').textContent = challenge.story;
    document.getElementById('challengeText').textContent = challenge.challenge;
    gameState.currentAnswer = challenge.answer;

    updateUI();
    clearCanvas();
}

function updateUI() {
    document.getElementById('currentLevel').textContent = gameState.level;
    document.getElementById('currentChallenge').textContent = gameState.currentChallenge + 1;
    document.getElementById('totalChallenges').textContent = gameState.totalChallenges;
    document.getElementById('score').textContent = gameState.score;
    document.getElementById('timeLeft').textContent = gameState.timeLeft;

    const progress = ((gameState.currentChallenge) / gameState.totalChallenges) * 100;
    document.getElementById('progressFill').style.width = progress + '%';

    updateAchievements();
}

function startTimer() {
    if (gameState.timer) clearInterval(gameState.timer);

    gameState.timer = setInterval(() => {
        gameState.timeLeft--;
        updateUI();

        if (gameState.timeLeft <= 0) {
            clearInterval(gameState.timer);
            completeGame();
        }
    }, 1000);
}

async function submitAnswer() {
    const dataURL = canvas.toDataURL('image/png');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ 
                image: dataURL,
                expected_answer: gameState.currentAnswer 
            })
        });
        const result = await response.json();

        showResult(result);
    } catch (error) {
        console.error('Prediction failed:', error);
    }
}

function showResult(result) {
    const resultBox = document.getElementById('resultBox');
    const resultText = document.getElementById('resultText');

    resultBox.classList.remove('hidden', 'correct', 'incorrect');

    if (result.correct) {
        resultBox.classList.add('correct');
        resultText.textContent = `🎉 Correct! You drew ${result.prediction}! ${result.feedback}`;
        gameState.score += result.points;

        // Add achievements
        if (result.achievements) {
            result.achievements.forEach(achievement => {
                gameState.achievements.add(achievement);
            });
        }
    } else {
        resultBox.classList.add('incorrect');
        resultText.textContent = `Oops! You drew ${result.prediction}, but I was looking for ${gameState.currentAnswer}. ${result.feedback}`;
    }

    document.getElementById('nextBtn').classList.remove('hidden');
    updateUI();
}

function nextChallenge() {
    gameState.currentChallenge++;
    document.getElementById('resultBox').classList.add('hidden');
    document.getElementById('nextBtn').classList.add('hidden');
    loadCurrentChallenge();
}

function completeGame() {
    clearInterval(gameState.timer);
    document.getElementById('gameInterface').classList.add('hidden');
    document.getElementById('gameComplete').classList.remove('hidden');

    document.getElementById('finalScore').innerHTML = `
        <h3>Final Score: ${gameState.score} points!</h3>
        <p>You completed ${gameState.currentChallenge} out of ${gameState.totalChallenges} challenges!</p>
    `;

    const achievementsList = Array.from(gameState.achievements).join(', ');
    document.getElementById('finalAchievements').innerHTML = `
        <h4>🏆 Achievements Unlocked:</h4>
        <p>${achievementsList || 'Keep practicing to unlock achievements!'}</p>
    `;
}

function updateAchievements() {
    const container = document.getElementById('achievements');
    container.innerHTML = '';

    gameState.achievements.forEach(achievement => {
        const badge = document.createElement('div');
        badge.className = 'achievement';
        badge.textContent = achievement;
        container.appendChild(badge);
    });
}

async function getHint() {
    try {
        const response = await fetch('/hint', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ 
                answer: gameState.currentAnswer,
                challenge: gameState.challenges[gameState.currentChallenge]
            })
        });
        const result = await response.json();

        alert(`💡 Hint: ${result.hint}`);
    } catch (error) {
        console.error('Failed to get hint:', error);
    }
}

function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('resultBox').classList.add('hidden');
    document.getElementById('nextBtn').classList.add('hidden');
}

function showLevelSelection() {
    document.getElementById('levelSelection').classList.remove('hidden');
    document.getElementById('gameInterface').classList.add('hidden');
    document.getElementById('gameComplete').classList.add('hidden');
}

// Event listeners
document.getElementById('clearBtn').addEventListener('click', clearCanvas);
document.getElementById('submitBtn').addEventListener('click', submitAnswer);
document.getElementById('hintBtn').addEventListener('click', getHint);
document.getElementById('nextBtn').addEventListener('click', nextChallenge);
document.getElementById('newGameBtn').addEventListener('click', showLevelSelection);

// Popup event listeners
document.getElementById('popupNextBtn').addEventListener('click', closePopupAndNext);
document.getElementById('popupTryAgainBtn').addEventListener('click', closePopupAndRetry);

document.getElementById('brushSize').addEventListener('input', (e) => {
    ctx.lineWidth = Number(e.target.value);
});

// Close popup when clicking outside (optional)
document.getElementById('resultPopup').addEventListener('click', (e) => {
    if (e.target.id === 'resultPopup') {
        // Don't close automatically - force user to click a button
        // closePopupAndNext();
    }
});

// Initialize
initCanvas();
</script>
</body>
</html>
"""


def preprocess_image_from_bytes(img_bytes):
    """Same preprocessing function as original"""
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    img = ImageOps.invert(img)

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    if img.size[0] == 0 or img.size[1] == 0:
        blank = Image.new('L', (28, 28), 0)
        return np.array(blank, dtype=np.float32).reshape(1, -1) / 255.0

    width, height = img.size
    max_side = max(width, height)
    new_width = int(round(width * 20.0 / max_side))
    new_height = int(round(height * 20.0 / max_side))
    img = img.resize((new_width, new_height), resample=Image.LANCZOS)

    new_img = Image.new('L', (28, 28), 0)
    upper_left = ((28 - new_width) // 2, (28 - new_height) // 2)
    new_img.paste(img, upper_left)

    arr = np.array(new_img, dtype=np.float32)
    if arr.sum() > 0:
        cy, cx = np.indices(arr.shape)
        total = arr.sum()
        x_center = (cx * arr).sum() / total
        y_center = (cy * arr).sum() / total
        shift_x = int(round(14 - x_center))
        shift_y = int(round(14 - y_center))
        new_img = ImageChops.offset(new_img, shift_x, shift_y)

    arr = np.array(new_img, dtype=np.float32)
    arr = arr / 255.0
    flat = arr.reshape(1, -1)
    return flat


def generate_challenges(level, count):
    """Generate challenges using Gemini AI"""
    level_config = LEVELS[level]
    min_num, max_num = level_config["range"]

    prompt = f"""
    Create {count} fun, educational challenges for kids learning numbers {min_num}-{max_num}.
    Each challenge should include:
    1. A short, engaging story context (2-3 sentences)
    2. A clear challenge asking them to draw a specific number
    3. The answer (which number to draw)

    Make it fun with themes like:
    - Animals and pets
    - Space adventures
    - Fairy tales
    - Superheroes
    - Pirates and treasure
    - Magic and wizards

    Return as JSON array with format:
    [{{
        "story": "story text",
        "challenge": "challenge text",
        "answer": number
    }}, ...]

    Make each challenge unique and age-appropriate for 5-8 year olds.
    """

    try:
        response = model.generate_content(prompt)
        # Extract JSON from response
        content = response.text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1]

        challenges = json.loads(content.strip())

        # Ensure answers are within range
        for challenge in challenges:
            if challenge["answer"] < min_num or challenge["answer"] > max_num:
                challenge["answer"] = random.randint(min_num, max_num)

        return challenges

    except Exception as e:
        print(f"Error generating challenges with Gemini: {e}")
        # Fallback challenges
        return generate_fallback_challenges(level, count)


def generate_fallback_challenges(level, count):
    """Fallback challenges if Gemini fails"""
    level_config = LEVELS[level]
    min_num, max_num = level_config["range"]

    themes = [
        ("🐶 Benny the dog found {} bones in the park!", "Draw the number of bones Benny found!"),
        ("🚀 Captain Space needs {} rockets for the mission!", "Draw the number of rockets needed!"),
        ("🎂 It's Emma's birthday and she's turning {} years old!", "Draw Emma's age number!"),
        ("🌟 The magic spell requires {} stars to work!", "Draw the number of magic stars!"),
        ("🏴‍☠️ The pirate treasure has {} golden coins!", "Draw the number of treasure coins!"),
    ]

    challenges = []
    for i in range(count):
        number = random.randint(min_num, max_num)
        story_template, challenge_template = random.choice(themes)

        challenges.append({
            "story": story_template.format(number),
            "challenge": challenge_template,
            "answer": number
        })

    return challenges


def generate_feedback(correct, prediction, expected, level):
    """Generate encouraging feedback using Gemini AI"""
    if correct:
        prompts = [
            "Great job! You drew that number perfectly!",
            "Awesome! You're getting really good at this!",
            "Excellent work! That's exactly right!",
            "Perfect! You're a number-drawing superstar!",
            "Amazing! You nailed it!",
        ]
        return random.choice(prompts)
    else:
        return f"Good try! The number {expected} is a bit different. Keep practicing!"


def calculate_achievements(session_data, prediction, expected, correct, time_taken):
    """Calculate achievements based on performance"""
    achievements = []

    if 'achievements' not in session_data:
        session_data['achievements'] = set()

    # First correct answer
    if correct and len(session_data.get('correct_answers', [])) == 0:
        achievements.append("🌟 First Success!")

    # Speed achievements
    if correct and time_taken < 10:
        achievements.append("⚡ Lightning Fast!")

    # Accuracy streak
    recent_answers = session_data.get('recent_answers', [])
    if len(recent_answers) >= 2 and all(recent_answers[-2:]):
        achievements.append("🎯 Accuracy Master!")

    # Number specific achievements
    if correct:
        if expected == 0:
            achievements.append("⭕ Zero Hero!")
        elif expected == 1:
            achievements.append("1️⃣ One and Done!")
        elif expected == 5:
            achievements.append("🖐️ High Five!")
        elif expected == 9:
            achievements.append("9️⃣ Nine Lives!")

    return achievements


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/start_game", methods=["POST"])
def start_game():
    data = request.get_json()
    level = data.get("level", "beginner")

    if level not in LEVELS:
        return jsonify({"error": "Invalid level"}), 400

    level_config = LEVELS[level]
    challenges = generate_challenges(level, level_config["challenges"])

    # Initialize session data
    session['game_start'] = datetime.now().isoformat()
    session['level'] = level
    session['challenges'] = challenges
    session['current_challenge'] = 0
    session['score'] = 0
    session['correct_answers'] = []
    session['recent_answers'] = []
    session['achievements'] = []

    return jsonify({
        "challenges": challenges,
        "total_challenges": len(challenges),
        "time_limit": level_config["time_limit"]
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    data_url = data["image"]
    expected_answer = data.get("expected_answer")

    if "," not in data_url:
        return jsonify({"error": "Invalid image data"}), 400

    header, b64 = data_url.split(",", 1)

    try:
        img_bytes = base64.b64decode(b64)
        x = preprocess_image_from_bytes(img_bytes)
        prediction = int(clf.predict(x)[0])

        # Check correctness
        correct = prediction == expected_answer

        # Calculate points
        points = 10 if correct else 0
        level = session.get('level', 'beginner')
        if level == 'intermediate':
            points *= 1.5
        elif level == 'advanced':
            points *= 2
        points = int(points)

        # Update session data
        if 'correct_answers' not in session:
            session['correct_answers'] = []
        if 'recent_answers' not in session:
            session['recent_answers'] = []

        session['correct_answers'].append(correct)
        session['recent_answers'].append(correct)
        if len(session['recent_answers']) > 5:
            session['recent_answers'].pop(0)

        # Calculate achievements
        time_taken = 30  # Placeholder - you'd track this properly
        achievements = calculate_achievements(session, prediction, expected_answer, correct, time_taken)

        # Generate feedback
        feedback = generate_feedback(correct, prediction, expected_answer, level)

        return jsonify({
            "prediction": prediction,
            "correct": correct,
            "points": points,
            "feedback": feedback,
            "achievements": achievements
        })

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/hint", methods=["POST"])
def get_hint():
    data = request.get_json()
    answer = data.get("answer")
    challenge = data.get("challenge", {})

    hint_prompts = {
        0: "Think of a big circle or oval shape!",
        1: "Draw a straight line from top to bottom!",
        2: "Start with a curve, then go across, then draw a line at the bottom!",
        3: "Make two curves that touch in the middle!",
        4: "Draw two lines that meet at the top like a tent!",
        5: "Start with a line across the top, then down, then a curve!",
        6: "Make a big curve that goes around and meets itself!",
        7: "Draw a line across the top, then a diagonal line down!",
        8: "Make two circles stacked on top of each other!",
        9: "Start with a circle at the top, then a straight line down!"
    }

    hint = hint_prompts.get(answer, "Take your time and think about the shape of this number!")

    return jsonify({"hint": hint})


if __name__ == "__main__":
    print("🎮 Starting Number Learning Adventure!")
    print("📝 Make sure to set your GEMINI_API_KEY environment variable")
    print("📁 Ensure 'svm_mnist_model.pkl' is in this folder")
    print("🌐 Game running at http://127.0.0.1:5000")
    app.run(debug=True)
