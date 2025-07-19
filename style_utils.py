import math
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay

# --- Normalized Modular Encoding ---
def normalized_modular(x):
    root = math.sqrt(x)
    frac = root - math.floor(root)
    return 2 * (frac - 0.5)

def encode_norm_mod(message):
    return [normalized_modular(ord(char)) for char in message if 32 <= ord(char) <= 126]  # basic filter

# --- Signal Features ---
def extract_signal_features(signal):
    signal = np.array(signal)
    if len(signal) < 2:
        return [0] * 8
    return [
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.median(signal),
        np.percentile(signal, 25),
        np.percentile(signal, 75),
        np.sum(np.abs(np.diff(signal))),
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
        np.mean(word_lengths) if word_lengths else 0,
        np.std(word_lengths) if word_lengths else 0,
        unique_word_count / max(word_count, 1),
        np.mean(sentence_lengths) if sentence_lengths else 0,
        punctuation_counts[','] / max(len(text), 1),
        punctuation_counts[';'] / max(len(text), 1),
        punctuation_counts['?'] / max(len(text), 1),
        punctuation_counts['.'] / max(len(text), 1),
    ] + function_word_freq

# --- Combined Features ---
def extract_combined_features(text):
    signal = encode_norm_mod(text)
    return extract_signal_features(signal) + extract_stylometric_features(text)

# --- Chunk Long Text ---
def chunk_text(text, chunk_size=300):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return [chunk for chunk in chunks if len(re.findall(r'\w', chunk)) > 50]

# --- PCA Plot ---
def plot_pca(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.7, edgecolors='k')
    ax.set_title("PCA: Author Style Clustering")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True)
    return fig

# --- Fingerprint Plot with Delaunay ---
def get_fingerprint_plot(X, y, author_label):
    data = X[y == author_label]
    if len(data) < 3:
        return blank_fig(f"Fingerprint - Author {author_label + 1} (insufficient data)")

    pca = PCA(n_components=2)
    coords = pca.fit_transform(data)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Fingerprint Pattern - Author {author_label + 1}")
    ax.scatter(coords[:, 0], coords[:, 1], c='blue' if author_label == 0 else 'red', alpha=0.7)

    try:
        tri = Delaunay(coords)
        for simplex in tri.simplices:
            pts = coords[simplex]
            ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], 'k-', lw=0.4, alpha=0.5)
            ax.plot([pts[1][0], pts[2][0]], [pts[1][1], pts[2][1]], 'k-', lw=0.4, alpha=0.5)
            ax.plot([pts[2][0], pts[0][0]], [pts[2][1], pts[0][1]], 'k-', lw=0.4, alpha=0.5)
    except Exception as e:
        print("Delaunay error:", e)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.grid(False)
    return fig

# --- Fallback Blank Figure ---
def blank_fig(title="No Plot"):
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, title, ha='center', va='center')
    ax.axis('off')
    return fig

# --- Author Fingerprint Vector ---
def get_author_fingerprint(X):
    return np.mean(X, axis=0) if len(X) else np.zeros(X.shape[1])
