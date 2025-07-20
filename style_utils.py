import math
import re
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from collections import Counter
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import base64

# --- Normalized Modular Function ---
def normalized_modular(x):
    root = math.sqrt(x)
    frac = root - math.floor(root)
    return 2 * (frac - 0.5)

def encode_norm_mod(message):
    return [normalized_modular(ord(char)) for char in message]

# --- Read Text from File ---
def read_text_file(file_storage):
    return file_storage.read().decode('utf-8', errors='ignore')

# --- Signal Features ---
def extract_signal_features(signal):
    signal = np.array(signal)
    return [
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.median(signal),
        np.percentile(signal, 25),
        np.percentile(signal, 75),
        np.sum(np.abs(np.diff(signal))),  # signal volatility
    ]

# --- Stylometric Features ---
FUNCTION_WORDS = {"the", "and", "of", "to", "is", "in", "it", "that", "i", "you", "a", "with", "for", "on"}

def extract_stylometric_features(text):
    words = re.findall(r'\b\w+\b', text.lower())
    word_lengths = [len(w) for w in words]
    sentences = re.split(r'[.!?]', text)
    sentence_lengths = [len(s.split()) for s in sentences if len(s.strip()) > 1]
    word_count = len(words)
    unique_word_count = len(set(words))
    punctuation_counts = Counter(re.findall(r'[.,;?!]', text))
    function_word_freq = [words.count(w) / max(word_count, 1) for w in FUNCTION_WORDS]

    return [
        np.mean(word_lengths),
        np.std(word_lengths),
        unique_word_count / max(word_count, 1),
        np.mean(sentence_lengths) if sentence_lengths else 0,
        punctuation_counts[','] / max(len(text), 1),
        punctuation_counts[';'] / max(len(text), 1),
        punctuation_counts['?'] / max(len(text), 1),
        punctuation_counts['.'] / max(len(text), 1),
    ] + function_word_freq

# --- Combine Features ---
def extract_combined_features(text):
    signal = encode_norm_mod(text)
    return extract_signal_features(signal) + extract_stylometric_features(text)

# --- Chunking ---
def chunk_text(text, chunk_size=300):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if len(text[i:i+chunk_size]) > 50]

# --- PCA Plot ---
def plot_pca(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', s=40, alpha=0.7, edgecolors='k')
    plt.title("PCA: Author Style Clustering")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# --- Fingerprint Plot ---
def plot_fingerprint(points, title="Author Fingerprint"):
    plt.figure(figsize=(5, 5))
    plt.scatter(points[:, 0], points[:, 1], c='dodgerblue', s=40, alpha=0.6, edgecolor='k')

    # Convex hull for outer boundary
    if len(points) >= 3:
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'red', lw=2)

    # Connect nearby inner points for fingerprint feel
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < np.percentile(np.linalg.norm(points - points[i], axis=1), 30):
                plt.plot([points[i, 0], points[j, 0]],
                         [points[i, 1], points[j, 1]],
                         color='lightgray', linewidth=0.6)

    plt.title(title)
    plt.axis('off')
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
