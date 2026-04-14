import os
import uuid
import sqlite3
import requests
import time

from werkzeug.utils import secure_filename

from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash


# ---------------- FILE VALIDATION ---------------- #

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------- APP SETUP ---------------- #

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret123'

# ✅ Increased file size (50MB)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# ---------------- DATABASE ---------------- #

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')

    conn.commit()
    conn.close()

init_db()


# ---------------- USER ---------------- #

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password


@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    c.execute("SELECT * FROM users WHERE id=?", (user_id,))
    user = c.fetchone()

    conn.close()

    if user:
        return User(user[0], user[1], user[2])
    return None


# ---------------- FILE PATH ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------- HF API ---------------- #

HF_API_URL = "https://shivanshuasthana81-deepfake-detector.hf.space/run/predict"


def predict_video_api(filepath):
    """
    Calls Hugging Face API
    Includes retry + safe parsing
    """

    for attempt in range(2):
        try:
            with open(filepath, "rb") as f:
                response = requests.post(
                    HF_API_URL,
                    files={
                        "data": (
                            os.path.basename(filepath),
                            f,
                            "video/mp4"
                        )
                    },
                    timeout=120
                )

            if response.status_code != 200:
                print("❌ API ERROR:", response.text)
                return "ERROR", 0

            result = response.json()

            print("🔍 RAW API RESPONSE:", result)  # DEBUG

            # ✅ SAFE PARSING
            if "data" not in result or len(result["data"]) < 2:
                return "ERROR", 0

            label = str(result["data"][0])

            try:
                confidence = float(result["data"][1])
            except:
                confidence = 0.0

            return label, round(confidence, 2)

        except requests.exceptions.Timeout:
            print("⏳ Timeout, retrying...")
            time.sleep(3)

        except Exception as e:
            print("❌ REQUEST ERROR:", e)
            return "ERROR", 0

    return "TIMEOUT (HF sleeping)", 0


# ---------------- ROUTES ---------------- #

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        try:
            c.execute("INSERT INTO users VALUES (?, ?, ?)",
                      (username, username, hashed_password))
            conn.commit()
            flash("Registered Successfully")
        except:
            flash("Username already exists")

        conn.close()
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()

        conn.close()

        if user and check_password_hash(user[2], password):
            login_user(User(user[0], user[1], user[2]))
            return redirect(url_for('dashboard'))

        flash("Invalid credentials")

    return render_template('login.html')


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if request.method == 'POST':

        if 'video' not in request.files:
            flash("No file uploaded")
            return redirect(url_for('dashboard'))

        file = request.files['video']

        if file.filename == '':
            flash("No file selected")
            return redirect(url_for('dashboard'))

        if not allowed_file(file.filename):
            flash("Invalid file format")
            return redirect(url_for('dashboard'))

        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(filepath)

        print("📤 Uploaded:", filepath)

        # ✅ CALL HF API
        label, confidence = predict_video_api(filepath)

        # ✅ DELETE FILE
        if os.path.exists(filepath):
            os.remove(filepath)

        return render_template(
            'result.html',
            label=label,
            confidence=confidence
        )

    return render_template('dashboard.html', username=current_user.username)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/health')
def health():
    return "OK", 200


@app.errorhandler(413)
def too_large(e):
    return "File too large (Max 50MB)", 413


# ---------------- MAIN ---------------- #

if __name__ == '__main__':
    app.run(debug=True)
