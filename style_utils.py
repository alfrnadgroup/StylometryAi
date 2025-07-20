import math
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix

# --- Normalized Modular Function ---
def normalized_modular(x):
    root = math.sqrt(x)
    frac = root - math.floor(root)
    return 2 * (frac - 0.5)

def encode_norm_mod(message):
    return [normalized_modular(ord(char)) for char in message]

# --- Read Text from File ---
def read_text_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

# --- Extract Signal Features ---
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
def plot_pca_clusters(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.7)
    ax.set_title("PCA Style Clusters")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True)
    return fig

# --- Fingerprint Visualization ---
def plot_fingerprint(vectors, title="Author"):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(vectors)
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(X_pca[:, 0], X_pca[:, 1], color='teal', s=30)

    # Connect all points to nearest neighbors (simulate fingerprint lines)
    dist_mat = distance_matrix(X_pca, X_pca)
    np.fill_diagonal(dist_mat, np.inf)
    for i in range(len(X_pca)):
        nearest_indices = np.argsort(dist_mat[i])[:3]  # Connect to 3 nearest
        for j in nearest_indices:
            ax.plot([X_pca[i, 0], X_pca[j, 0]], [X_pca[i, 1], X_pca[j, 1]], 'lightgray', linewidth=1)

    ax.set_title(f"{title}'s Stylometric Fingerprint")
    ax.axis('off')
    return fig
