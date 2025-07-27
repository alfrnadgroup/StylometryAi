from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
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
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import os
import jwt
import random
import datetime

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

SECRET_KEY = "your_secret_key_here"
otp_store = {}

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.route("/")
def serve_index():
    return app.send_static_file("index.html")

@app.route("/api/request-otp", methods=["POST"])
def request_otp():
    data = request.json
    email = data.get("email")
    if not email:
        return jsonify({"error": "Email required"}), 400

    otp = str(random.randint(100000, 999999))
    otp_store[email] = otp
    print(f"[DEBUG] OTP for {email} is {otp}")
    return jsonify({"message": "OTP sent to your email (check console for testing)"})

@app.route("/api/verify-otp", methods=["POST"])
def verify_otp():
    data = request.json
    email = data.get("email")
    otp = data.get("otp")

    if not email or not otp:
        return jsonify({"error": "Email and OTP required"}), 400

    if otp_store.get(email) != otp:
        return jsonify({"error": "Invalid OTP"}), 401

    payload = {
        "email": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    del otp_store[email]
    return jsonify({"token": token})

@app.route("/analyze", methods=["POST"])
def analyze():
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing or invalid token"}), 403

    token = auth_header.split(" ")[1]
    try:
        jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return jsonify({"error": "Token expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid token"}), 401

    file1 = request.files.get("author1")
    file2 = request.files.get("author2")

    if not file1 or not file2:
        return jsonify({"error": "Please upload both files."}), 400

    text1 = file1.read().decode("utf-8")
    text2 = file2.read().decode("utf-8")

    chunks1 = chunk_text(text1)
    chunks2 = chunk_text(text2)

    features, labels = [], []
    for chunk in chunks1:
        features.append(extract_combined_features(chunk))
        labels.append(0)
    for chunk in chunks2:
        features.append(extract_combined_features(chunk))
        labels.append(1)

    X, y = np.array(features), np.array(labels)
    if len(set(y)) < 2:
        return jsonify({"error": "Not enough variation to distinguish authors."}), 400

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = round(clf.score(X_test, y_test) * 100, 2)
    report = classification_report(y_test, y_pred, target_names=["Author1", "Author2"])

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
