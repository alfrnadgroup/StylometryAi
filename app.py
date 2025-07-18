from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from style_utils import (
    extract_combined_features,
    chunk_text,
    plot_pca,
    get_fingerprint_plot
)

app = Flask(__name__, static_folder="static")

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route("/")
def serve_index():
    return send_from_directory("static", "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file1 = request.files["author1"]
    file2 = request.files["author2"]

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
        return jsonify({"error": "Not enough data for both authors."}), 400

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = clf.score(X_test, y_test)

    # 📌 Verdict
    if accuracy > 0.85:
        verdict = "✅ Likely DIFFERENT authors (high confidence)."
    elif accuracy < 0.6:
        verdict = "⚠️ Possibly SAME author or very similar style."
    else:
        verdict = "❓ Unclear result. Consider larger or clearer samples."

    # 📈 Visuals
    pca_fig = plot_pca(X, y)
    fp1_fig = get_fingerprint_plot(X, y, author_label=0)
    fp2_fig = get_fingerprint_plot(X, y, author_label=1)

    return jsonify({
        "accuracy": round(accuracy * 100, 2),
        "verdict": verdict,
        "report": classification_report(y_test, y_pred, target_names=["Author1", "Author2"]),
        "fingerprint1": np.round(np.mean(X[y == 0], axis=0), 3).tolist(),
        "fingerprint2": np.round(np.mean(X[y == 1], axis=0), 3).tolist(),
        "pca_plot": fig_to_base64(pca_fig),
        "fp1_plot": fig_to_base64(fp1_fig),
        "fp2_plot": fig_to_base64(fp2_fig)
    })

if __name__ == "__main__":
    app.run(debug=True)
