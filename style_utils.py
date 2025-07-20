import math
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay

# --- Normalized Modular Function ---
def normalized_modular(x):
    root = math.sqrt(x)
    frac = root - math.floor(root)
    return 2 * (frac - 0.5)

def encode_norm_mod(message):
    return [normalized_modular(ord(char)) for char in message]

# --- Extract NormMod Signal Features ---
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

# --- Chunk Text ---
def chunk_text(text, chunk_size=300):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if len(text[i:i+chunk_size]) > 50]

# --- PCA Visualization ---
def plot_pca(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.7, edgecolors='k')
    ax.set_title("PCA: Author Style Clustering")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(True)
    return fig

# --- Fingerprint Visualization ---
def get_fingerprint_plot(X, y, author_label):
    data = X[y == author_label]
    pca = PCA(n_components=2)
    coords = pca.fit_transform(data)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Fingerprint Pattern - Author {author_label + 1}")
    ax.scatter(coords[:, 0], coords[:, 1], c='blue' if author_label == 0 else 'red', alpha=0.7)

    # Connect nearby points using Delaunay triangulation
    try:
        tri = Delaunay(coords)
        for simplex in tri.simplices:
            for i in range(3):
                x0, y0 = coords[simplex[i]]
                x1, y1 = coords[simplex[(i + 1) % 3]]
                ax.plot([x0, x1], [y0, y1], 'k-', lw=0.4, alpha=0.5)
    except:
        pass  # triangulation may fail on tiny sets

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.grid(False)
    return fig
