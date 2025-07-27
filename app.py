from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import jwt
import json
import random
import string
from datetime import datetime, timedelta, timezone
import requests
from dotenv import load_dotenv
import base64
import io
from stylometry_engine import analyze_texts  # your core analysis logic

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Environment variables
JWT_SECRET = os.getenv("JWT_SECRET", "your-jwt-secret")
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")

JWT_ALGORITHM = "HS256"
OTP_EXPIRY_MINUTES = 5
JWT_EXPIRY_HOURS = 1
otp_store = {}
OTP_FILE = "otp_store.json"

# ---------- Utility Functions ----------

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

def generate_jwt(email):
    return jwt.encode(
        {
            "email": email,
            "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS)
        },
        JWT_SECRET,
        algorithm=JWT_ALGORITHM
    )

def validate_jwt_token():
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def load_otp_store():
    if os.path.exists(OTP_FILE):
        with open(OTP_FILE) as f:
            raw = json.load(f)
            for email, v in raw.items():
                otp_store[email] = {
                    "otp": v["otp"],
                    "expires": datetime.fromisoformat(v["expires"])
                }

def save_otp_store():
    with open(OTP_FILE, "w") as f:
        json.dump({
            email: {
                "otp": v["otp"],
                "expires": v["expires"].isoformat()
            } for email, v in otp_store.items()
        }, f)

load_otp_store()

# ---------- OTP API Routes ----------

@app.route("/api/request-otp", methods=["POST"])
def request_otp():
    data = request.get_json()
    email = data.get("email")
    if not email:
        return jsonify({"error": "Email is required"}), 400

    otp = generate_otp()
    expiry = datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRY_MINUTES)
    otp_store[email] = {"otp": otp, "expires": expiry}
    save_otp_store()

    # Send OTP via Brevo
    payload = {
        "sender": {"name": "StylometryAI", "email": SENDER_EMAIL},
        "to": [{"email": email}],
        "subject": "Your StylometryAI OTP Code",
        "htmlContent": f"<p>Your OTP code is <strong>{otp}</strong>. It expires in 5 minutes.</p>"
    }
    headers = {
        "api-key": BREVO_API_KEY,
        "Content-Type": "application/json"
    }
    try:
        res = requests.post("https://api.brevo.com/v3/smtp/email", json=payload, headers=headers)
        if res.status_code == 201:
            return jsonify({"message": "OTP sent"})
        return jsonify({"error": "Failed to send OTP"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/verify-otp", methods=["POST"])
def verify_otp():
    data = request.get_json()
    email = data.get("email")
    otp = data.get("otp")

    if not email or not otp:
        return jsonify({"error": "Email and OTP are required"}), 400

    record = otp_store.get(email)
    if not record:
        return jsonify({"error": "No OTP request found"}), 400

    if datetime.now(timezone.utc) > record["expires"]:
        del otp_store[email]
        save_otp_store()
        return jsonify({"error": "OTP expired"}), 400

    if record["otp"] != otp:
        return jsonify({"error": "Incorrect OTP"}), 400

    # Success
    del otp_store[email]
    save_otp_store()
    token = generate_jwt(email)
    return jsonify({"message": "OTP verified", "token": token})

# ---------- Protected Analysis Route ----------

@app.route("/analyze", methods=["POST"])
def analyze():
    payload = validate_jwt_token()
    if not payload:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        file1 = request.files["author1"]
        file2 = request.files["author2"]

        text1 = file1.read().decode("utf-8")
        text2 = file2.read().decode("utf-8")

        result = analyze_texts(text1, text2)  # <- your stylometry_engine function

        # Return all analysis data including plots as base64
        return jsonify({
            "accuracy": result["accuracy"],
            "report": result["report"],
            "fingerprint1": result["fingerprint1"],
            "fingerprint2": result["fingerprint2"],
            "fp1_plot": result["fp1_plot_base64"],
            "fp2_plot": result["fp2_plot_base64"],
            "pca_plot": result["pca_plot_base64"]
        })
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

# ---------- Run App ----------
if __name__ == "__main__":
    app.run(debug=True)
