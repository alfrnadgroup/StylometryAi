from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, io, base64, json, random, string, time, jwt, requests
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import numpy as np

from style_utils import (
    extract_combined_features,
    chunk_text,
    plot_pca,
    get_fingerprint_plot,
    get_author_fingerprint
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dotenv import load_dotenv

# --- Config ---
load_dotenv()
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

JWT_SECRET = os.getenv("JWT_SECRET", "secret")
JWT_ALGORITHM = 'HS256'
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "noreply@example.com")
OTP_FILE = "otp_store.json"
otp_store = {}

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# --- JWT ---
def generate_jwt(email):
    payload = {
        'email': email,
        'iat': time.time(),
        'exp': time.time() + 3600
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def validate_token():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None, jsonify({'message': 'Unauthorized'}), 401
    token = auth_header.split()[1]
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM]), None, None
    except jwt.ExpiredSignatureError:
        return None, jsonify({'message': 'Token expired'}), 401
    except Exception:
        return None, jsonify({'message': 'Invalid token'}), 401

# --- OTP handling ---
def load_otp_store():
    if os.path.exists(OTP_FILE):
        with open(OTP_FILE, "r") as f:
            data = json.load(f)
            for email, entry in data.items():
                otp_store[email] = {
                    "otp": entry["otp"],
                    "expires": datetime.fromisoformat(entry["expires"])
                }

def save_otp_store():
    with open(OTP_FILE, "w") as f:
        json.dump({
            email: {
                "otp": val["otp"],
                "expires": val["expires"].isoformat()
            } for email, val in otp_store.items()
        }, f)

load_otp_store()

@app.route("/")
def serve_frontend():
    return app.send_static_file("index.html")

@app.route("/api/request-otp", methods=["POST"])
def request_otp():
    data = request.json
    email = data.get("email")
    if not email:
        return jsonify({"error": "Email required"}), 400

    otp = ''.join(random.choices(string.digits, k=6))
    expiry = datetime.now(timezone.utc) + timedelta(minutes=5)
    otp_store[email] = {"otp": otp, "expires": expiry}
    save_otp_store()

    headers = {
        "api-key": BREVO_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "sender": {"name": "Stylometry AI", "email": SENDER_EMAIL},
        "to": [{"email": email}],
        "subject": "Your OTP Code",
        "htmlContent": f"<p>Your OTP is: <strong>{otp}</strong></p>"
    }

    try:
        r = requests.post(BREVO_API_URL, json=payload, headers=headers)
        if r.status_code == 201:
            return jsonify({"message": "OTP sent"}), 200
        else:
            return jsonify({"error": "Email service failed"}), 500
    except Exception:
        return jsonify({"error": "Failed to send OTP"}), 500

@app.route("/api/verify-otp", methods=["POST"])
def verify_otp():
    data = request.json
    email, otp_input = data.get("email"), data.get("otp")
    if not email or not otp_input:
        return jsonify({"error": "Email and OTP required"}), 400

    stored = otp_store.get(email)
    if not stored:
        return jsonify({"error": "OTP not requested"}), 400
    if datetime.now(timezone.utc) > stored["expires"]:
        del otp_store[email]
        save_otp_store()
        return jsonify({"error": "OTP expired"}), 400
    if stored["otp"] != otp_input:
        return jsonify({"error": "Invalid OTP"}), 400

    del otp_store[email]
    save_otp_store()
    token = generate_jwt(email)
    return jsonify({"message": "OTP verified", "token": token})

@app.route("/api/analyze", methods=["POST"])
def analyze():
    payload, err, code = validate_token()
    if err: return err, code

    f1 = request.files.get("author1")
    f2 = request.files.get("author2")
    if not f1 or not f2:
        return jsonify({"error": "Both files required"}), 400

    try:
        t1 = f1.read().decode("utf-8")
        t2 = f2.read().decode("utf-8")
    except Exception:
        return jsonify({"error": "Invalid file format"}), 400

    chunks1 = chunk_text(t1)
    chunks2 = chunk_text(t2)

    if len(chunks1) < 2 or len(chunks2) < 2:
        return jsonify({"error": "Not enough text"}), 400

    X, y = [], []
    for c in chunks1: X.append(extract_combined_features(c)); y.append(0)
    for c in chunks2: X.append(extract_combined_features(c)); y.append(1)
    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    acc = round(model.score(X_test, y_test) * 100, 2)
    report = classification_report(y_test, model.predict(X_test), target_names=["Author1", "Author2"])

    pca_fig = plot_pca(X, y)
    fp1_fig = get_fingerprint_plot(X, y, 0)
    fp2_fig = get_fingerprint_plot(X, y, 1)
    fp1 = get_author_fingerprint(X[y == 0]).tolist()
    fp2 = get_author_fingerprint(X[y == 1]).tolist()

    return jsonify({
        "accuracy": acc,
        "report": report,
        "fingerprint1": fp1,
        "fingerprint2": fp2,
        "pca_plot": fig_to_base64(pca_fig),
        "fp1_plot": fig_to_base64(fp1_fig),
        "fp2_plot": fig_to_base64(fp2_fig),
    })

if __name__ == "__main__":
    app.run(debug=True)
