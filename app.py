from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from style_utils import (
    extract_combined_features,
    chunk_text,
    plot_pca,
    get_fingerprint_plot,
    get_author_fingerprint
)
import numpy as np
import io
import base64
import os
import matplotlib.pyplot as plt
import json
import jwt
import random
import string
from datetime import datetime, timedelta, timezone
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# --- OTP + JWT CONFIG ---
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")
JWT_ALGORITHM = 'HS256'
OTP_FILE = 'otp_store.json'
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "your-sender@example.com")
BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"

otp_store = {}

def save_otp_store():
    with open(OTP_FILE, 'w') as f:
        json.dump({
            email: {'otp': v['otp'], 'expires': v['expires'].isoformat()}
            for email, v in otp_store.items()
        }, f)

def load_otp_store():
    if os.path.exists(OTP_FILE):
        with open(OTP_FILE) as f:
            raw = json.load(f)
            for email, v in raw.items():
                otp_store[email] = {
                    'otp': v['otp'],
                    'expires': datetime.fromisoformat(v['expires'])
                }

def generate_jwt(email):
    payload = {
        'email': email,
        'exp': datetime.now(timezone.utc) + timedelta(hours=1)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def validate_token():
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        return None
    try:
        return jwt.decode(auth.split()[1], JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except:
        return None

load_otp_store()

@app.route("/api/request-otp", methods=["POST"])
def request_otp():
    data = request.get_json()
    email = data.get("email")
    if not email:
        return jsonify({"error": "Email is required"}), 400

    otp = ''.join(random.choices(string.digits, k=6))
    expiry = datetime.now(timezone.utc) + timedelta(minutes=5)
    otp_store[email] = {'otp': otp, 'expires': expiry}
    save_otp_store()

    payload = {
        "sender": {"name": "Stylometry AI", "email": SENDER_EMAIL},
        "to": [{"email": email}],
        "subject": "Your OTP Code",
        "htmlContent": f"<p>Your OTP is <strong>{otp}</strong>. It expires in 5 minutes.</p>"
    }
    headers = {
        "api-key": BREVO_API_KEY,
        "Content-Type": "application/json"
    }
    try:
        res = requests.post(BREVO_API_URL, json=payload, headers=headers)
        if res.status_code == 201:
            return jsonify({"message": "OTP sent"}), 200
        return jsonify({"error": "Failed to send OTP"}), 500
    except:
        return jsonify({"error": "OTP service error"}), 500

@app.route("/api/verify-otp", methods=["POST"])
def verify_otp():
    data = request.get_json()
    email = data.get("email")
    otp = data.get("otp")

    stored = otp_store.get(email)
    if not email or not otp or not stored:
        return jsonify({"error": "Invalid email or OTP"}), 400
    if datetime.now(timezone.utc) > stored["expires"]:
        del otp_store[email]
        save_otp_store()
        return jsonify({"error": "OTP expired"}), 400
    if stored["otp"] != otp:
        return jsonify({"error": "Incorrect OTP"}), 400

    del otp_store[email]
    save_otp_store()
    token = generate_jwt(email)
    return jsonify({"message": "OTP verified", "token": token}), 200

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.route("/analyze", methods=["POST"])
def analyze():
    if not validate_token():
        return jsonify({"error": "Unauthorized"}), 401

    file1 = request.files.get("author1")
    file2 = request.files.get("author2")
    if not file1 or not file2:
        return jsonify({"error": "Please upload both files."}), 400

    text1 = file1.read().decode("utf-8")
    text2 = file2.read().decode("utf-8")
    chunks1 = chunk_text(text1)
    chunks2 = chunk_text(text2)

    features = [extract_combined_features(c) for c in chunks1 + chunks2]
    labels = [0]*len(chunks1) + [1]*len(chunks2)
    X = np.array(features)
    y = np.array(labels)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    acc = round(model.score(X_test, y_test)*100, 2)
    report = classification_report(y_test, model.predict(X_test), target_names=["Author1", "Author2"])

    return jsonify({
        "accuracy": acc,
        "report": report,
        "fingerprint1": get_author_fingerprint(X[y == 0]).tolist(),
        "fingerprint2": get_author_fingerprint(X[y == 1]).tolist(),
        "pca_plot": fig_to_base64(plot_pca(X, y)),
        "fp1_plot": fig_to_base64(get_fingerprint_plot(X, y, 0)),
        "fp2_plot": fig_to_base64(get_fingerprint_plot(X, y, 1)),
    })

@app.route("/")
def index():
    return app.send_static_file("stylometry.html")
