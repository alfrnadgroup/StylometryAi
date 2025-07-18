# style_utils.py

import math
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

FUNCTION_WORDS = {"the", "and", "of", "to", "is", "in", "it", "that", "i", "you", "a", "with", "for", "on"}

def normalized_modular(x):
    root = math.sqrt(x)
    frac = root - math.floor(root)
    return 2 * (frac - 0.5)

def encode_norm_mod(message):
    return [normalized_modular(ord(char)) for char in message]

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
        np.sum(np.abs(np.diff(signal))),
    ]

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

def extract_combined_features(text):
    signal = encode_norm_mod(text)
    return extract_signal_features(signal) + extract_stylometric_features(text)

def chunk_text(text, chunk_size=300):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if len(text[i:i+chunk_size]) > 50]

def get_author_fingerprint(vectors):
    return np.mean(vectors, axis=0)

def get_pca_plot(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.7, edgecolors='k')
    ax.set_title("PCA: Author Style Clustering")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True)
    return fig

def get_fingerprint_plot(X, y, author_index):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    points = X_pca[y == author_index]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(points[:, 0], points[:, 1], c='skyblue', edgecolors='k', alpha=0.6, label=f"Author {author_index+1}")

    if len(points) >= 3:
        hull = ConvexHull(points)
        polygon = Polygon(points[hull.vertices], closed=True, fill=True, edgecolor='black', facecolor='lightblue', alpha=0.3)
        ax.add_patch(polygon)
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], 'k-', linewidth=1)

    ax.set_title(f"Stylometric Fingerprint - Author {author_index+1}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True)
    ax.legend()
    return fig
