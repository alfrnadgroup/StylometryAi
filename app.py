from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import os
import base64
import matplotlib.pyplot as plt
import numpy as np
from style_utils import (
    extract_combined_features,
    chunk_text,
    plot_pca,
    get_fingerprint_plot
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

@app.route("/")
def serve_index():
    return app.send_static_file("index.html")

@app.route("/api/analyze", methods=["POST"])  # 🔄 Updated route
def analyze():
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
    app.run(host="0.0.0.0", port=port)
