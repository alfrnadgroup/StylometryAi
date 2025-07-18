import os
import io
import base64
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from style_utils import (
    read_text_file,
    chunk_text,
    extract_combined_features,
    plot_pca_clusters,
    plot_fingerprint
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file1 = request.files.get("author1")
    file2 = request.files.get("author2")

    if not file1 or not file2:
        return jsonify({"error": "Both files are required."}), 400

    path1 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file1.filename))
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file2.filename))
    file1.save(path1)
    file2.save(path2)

    text1 = read_text_file(path1)
    text2 = read_text_file(path2)

    chunks1 = chunk_text(text1)
    chunks2 = chunk_text(text2)

    if len(chunks1) < 2 or len(chunks2) < 2:
        return jsonify({"error": "Each file needs to contain at least two valid text chunks."}), 400

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

    report = classification_report(y_test, y_pred, target_names=["Author 1", "Author 2"])
    accuracy = round(clf.score(X_test, y_test) * 100, 2)

    if accuracy > 85:
        verdict = "🧠 Likely DIFFERENT authors (high confidence)."
    elif accuracy < 60:
        verdict = "🤔 Possibly SAME author or very similar style."
    else:
        verdict = "⚠️ Unclear result. Consider larger or clearer samples."

    # Fingerprint vector (mean of features)
    fp1 = np.mean([extract_combined_features(c) for c in chunks1], axis=0).tolist()
    fp2 = np.mean([extract_combined_features(c) for c in chunks2], axis=0).tolist()

    # Generate PCA cluster plot
    pca_img = plot_to_base64(plot_pca_clusters, X, y)

    # Fingerprint plots
    fp1_img = plot_to_base64(plot_fingerprint, [extract_combined_features(c) for c in chunks1], title="Author 1")
    fp2_img = plot_to_base64(plot_fingerprint, [extract_combined_features(c) for c in chunks2], title="Author 2")

    return jsonify({
        "accuracy": accuracy,
        "verdict": verdict,
        "report": report,
        "fingerprint1": [round(v, 3) for v in fp1],
        "fingerprint2": [round(v, 3) for v in fp2],
        "pca_plot": pca_img,
        "fp1_plot": fp1_img,
        "fp2_plot": fp2_img
    })


def plot_to_base64(plot_func, *args, **kwargs):
    """Capture a plot into a base64 image string."""
    fig = plot_func(*args, **kwargs)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
