# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from style_utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

@app.route("/")
def serve_index():
    return app.send_static_file("index.html")

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.route("/analyze", methods=["POST"])
def analyze():
    file1 = request.files.get("author1")
    file2 = request.files.get("author2")

    if not file1 or not file2:
        return jsonify({"error": "Both files are required."}), 400

    text1 = file1.read().decode("utf-8")
    text2 = file2.read().decode("utf-8")

    chunks1 = chunk_text(text1)
    chunks2 = chunk_text(text2)

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

    if len(set(y)) < 2 or len(y) < 4:
        return jsonify({"error": "Not enough data to distinguish authors."}), 400

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = round(clf.score(X_test, y_test) * 100, 2)

    fp1 = get_author_fingerprint(X[y == 0]).tolist()
    fp2 = get_author_fingerprint(X[y == 1]).tolist()
    report = classification_report(y_test, y_pred, target_names=["Author1", "Author2"])

    # Generate plots
    pca_fig = get_pca_plot(X, y)
    fp1_fig = get_fingerprint_plot(X, y, 0)
    fp2_fig = get_fingerprint_plot(X, y, 1)

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
