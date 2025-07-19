from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import io
import os
import base64
import matplotlib.pyplot as plt
from style_utils import (
    extract_combined_features,
    chunk_text,
    plot_pca,
    get_fingerprint_plot
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import random
import string
import time
import jwt
from datetime import datetime, timedelta, timezone
import requests
import json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# --- OTP and Auth Config ---
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")
JWT_ALGORITHM = 'HS256'
REGISTRATION_TOKEN = os.getenv("REGISTRATION_TOKEN", "REGISTER2025")
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "your-sender@example.com")
OTP_FILE = 'otp_store.json'

users = {
    'admin': {'password': 'adminpass', 'role': 'admin', 'email': 'admin@example.com'}
}

otp_store = {}

def load_otp_store():
    if os.path.exists(OTP_FILE):
        try:
            with open(OTP_FILE, 'r') as f:
                data = json.load(f)
                for email, item in data.items():
                    otp_store[email] = {
                        'otp': item['otp'],
                        'expires': datetime.fromisoformat(item['expires'])
                    }
        except Exception as e:
            print("Failed to load OTP store:", e)

def save_otp_store():
    try:
        data = {
            email: {'otp': val['otp'], 'expires': val['expires'].isoformat()}
            for email, val in otp_store.items()
        }
        with open(OTP_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print("Failed to save OTP store:", e)

load_otp_store()

def generate_jwt(username, role):
    payload = {
        'username': username,
        'role': role,
        'iat': int(time.time()),
        'exp': int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp())
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def validate_token():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None, jsonify({'message': 'Unauthorized'}), 401

    token = auth_header.split()[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload, None, None
    except jwt.ExpiredSignatureError:
        return None, jsonify({'message': 'Token expired'}), 401
    except Exception:
        return None, jsonify({'message': 'Invalid token'}), 401

# --- Routes ---

@app.route("/")
def serve_index():
    return app.send_static_file("index.html")

# OTP request endpoint
@app.route('/api/request-otp', methods=['POST'])
def request_otp():
    data = request.json
    email = data.get('email')

    if not email:
        return jsonify({'error': 'Email is required'}), 400

    otp = ''.join(random.choices(string.digits, k=6))
    expiry = datetime.now(timezone.utc) + timedelta(minutes=5)
    otp_store[email] = {'otp': otp, 'expires': expiry}
    save_otp_store()

    payload = {
        "sender": {"name": "Your App", "email": SENDER_EMAIL},
        "to": [{"email": email}],
        "subject": "Your OTP Code",
        "htmlContent": f"<p>Your OTP is <strong>{otp}</strong>. It expires in 5 minutes.</p>"
    }
    headers = {
        "api-key": BREVO_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        response = requests.post(BREVO_API_URL, json=payload, headers=headers)
        if response.status_code == 201:
            return jsonify({'message': 'OTP sent successfully.'}), 200
        else:
            print("Email error:", response.status_code, response.text)
            return jsonify({'error': 'Failed to send OTP.'}), 500
    except Exception as e:
        print("Request failed:", str(e))
        return jsonify({'error': 'Email service error'}), 500

# OTP verification endpoint
@app.route('/api/verify-otp', methods=['POST'])
def verify_otp():
    data = request.json
    email = data.get('email')
    user_otp = data.get('otp')

    if not email or not user_otp:
        return jsonify({'error': 'Email and OTP are required'}), 400

    if email not in otp_store:
        return jsonify({'error': 'OTP not requested for this email'}), 400

    stored = otp_store[email]
    if datetime.now(timezone.utc) > stored['expires']:
        del otp_store[email]
        save_otp_store()
        return jsonify({'error': 'OTP expired'}), 400

    if user_otp == stored['otp']:
        del otp_store[email]
        save_otp_store()
        return jsonify({'message': 'OTP verified successfully.'}), 200
    else:
        return jsonify({'error': 'Invalid OTP'}), 400

# Registration endpoint
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    role = data.get('role')
    email = data.get('email')
    otp = data.get('otp')
    reg_token = data.get('reg_token')

    if not all([username, password, role, email, otp, reg_token]):
        return jsonify({'message': 'Missing fields'}), 400

    if reg_token != REGISTRATION_TOKEN:
        return jsonify({'message': 'Invalid registration token'}), 403

    if username in users:
        return jsonify({'message': 'Username already exists'}), 400

    if email not in otp_store:
        return jsonify({'message': 'OTP not requested for this email'}), 400

    stored = otp_store[email]
    if datetime.now(timezone.utc) > stored['expires']:
        del otp_store[email]
        save_otp_store()
        return jsonify({'message': 'OTP expired'}), 400

    if otp != stored['otp']:
        return jsonify({'message': 'Invalid OTP'}), 400

    users[username] = {'password': password, 'role': role, 'email': email}
    del otp_store[email]
    save_otp_store()
    return jsonify({'message': 'User registered successfully'}), 200

# Login endpoint
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    role = data.get('role')

    user = users.get(username)
    if not user or user['password'] != password or user['role'] != role:
        return jsonify({'message': 'Invalid credentials or role'}), 401

    token = generate_jwt(username, role)
    return jsonify({'token': token, 'role': role})

# Authorized names endpoint (example protected)
@app.route('/api/authorized-names', methods=['GET'])
def get_authorized_names():
    payload, error, status = validate_token()
    if error:
        return error, status

    authorized_names = ['Alice Johnson', 'Bob Smith', 'Carol Lee']
    return jsonify({'names': authorized_names})

# Style analysis endpoint with file upload
@app.route("/analyze", methods=["POST"])
def analyze():
    # Check auth token for security (optional, add if needed)
    payload, error, status = validate_token()
    if error:
        return error, status

    file1 = request.files.get("author1")
    file2 = request.files.get("author2")

    if not file1 or not file2:
        return jsonify({"error": "Both files are required."}), 400

    text1 = file1.read().decode("utf-8")
    text2 = file2.read().decode("utf-8")

    chunks1 = chunk_text(text1)
    chunks2 = chunk_text(text2)

    if len(chunks1) < 2 or len(chunks2) < 2:
        return jsonify({"error": "Not enough text in one or both files."}), 400

    features = []
    labels = []

    for chunk in chunks1:
        features.append(extract_combined_features(chunk))
        labels.append(0)
    for chunk in chunks2:
        features.append(extract_combined_features(chunk))
        labels.append(1)

    X = np.array(features)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = round(clf.score(X_test, y_test) * 100, 2)
    report = classification_report(y_test, y_pred, target_names=["Author1", "Author2"])

    # Generate plots
    fig_pca = plot_pca(X, y)
    fig_fp1 = get_fingerprint_plot(X, y, author_label=0)
    fig_fp2 = get_fingerprint_plot(X, y, author_label=1)

    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({
        "accuracy": accuracy,
        "report": report,
        "fingerprint1": extract_combined_features(" ".join(chunks1[:2])),
        "fingerprint2": extract_combined_features(" ".join(chunks2[:2])),
        "pca_plot": fig_to_base64(fig_pca),
        "fp1_plot": fig_to_base64(fig_fp1),
        "fp2_plot": fig_to_base64(fig_fp2),
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
