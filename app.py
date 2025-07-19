from flask import Flask, request, jsonify, send_from_directory
from style_utils import (
    read_text_file,
    extract_combined_features,
    chunk_text,
    plot_pca,               # ✅ using 'plot_pca' instead of 'plot_pca_clusters'
    plot_fingerprint
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import base64
import io
import os

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file1 = request.files.get('author1')
    file2 = request.files.get('author2')

    if not file1 or not file2:
        return jsonify({"error": "Both files are required"}), 400

    text1 = file1.read().decode('utf-8')
    text2 = file2.read().decode('utf-8')

    chunks1 = chunk_text(text1)
    chunks2 = chunk_text(text2)

    if len(chunks1) < 2 or len(chunks2) < 2:
        return jsonify({"error": "Not enough text in one or both files."}), 400

    X = []
    y = []

    for chunk in chunks1:
        X.append(extract_combined_features(chunk))
        y.append(0)
    for chunk in chunks2:
        X.append(extract_combined_features(chunk))
        y.append(1)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=["Author1", "Author2"])
    accuracy = round(clf.score(X_test, y_test) * 100, 2)

    if accuracy > 85:
        verdict = "Likely DIFFERENT authors (high confidence)."
    elif accuracy < 60:
        verdict = "Possibly SAME author or very similar style."
    else:
        verdict = "Unclear result. Consider more or clearer samples."

    # --- PCA Plot
    pca_img = plot_pca(X, y)
    pca_b64 = base64.b64encode(pca_img.getvalue()).decode()

    # --- Fingerprints
    fp1_img = plot_fingerprint(X[y == 0])
    fp2_img = plot_fingerprint(X[y == 1])
    fp1_b64 = base64.b64encode(fp1_img.getvalue()).decode()
    fp2_b64 = base64.b64encode(fp2_img.getvalue()).decode()

    return jsonify({
        "accuracy": accuracy,
        "report": report,
        "verdict": verdict,
        "fingerprint1": X[y == 0][0].tolist(),
        "fingerprint2": X[y == 1][0].tolist(),
        "pca_plot": pca_b64,
        "fp1_plot": fp1_b64,
        "fp2_plot": fp2_b64
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
