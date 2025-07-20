import numpy as np
import matplotlib.pyplot as plt
import io
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
from itertools import combinations
import math

def read_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def chunk_text(text, chunk_size=1000):
    """Split the text into fixed-length chunks"""
    text = re.sub(r'\s+', ' ', text.strip())
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if len(text[i:i+chunk_size]) > 200]

def extract_combined_features(text):
    # Very basic stylometric features
    features = []
    words = re.findall(r'\b\w+\b', text.lower())
    sentences = re.split(r'[.!?]+', text)
    char_count = len(text)
    word_count = len(words)
    sentence_count = len(sentences)
    
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    avg_sentence_length = word_count / sentence_count if sentence_count else 0
    lexical_diversity = len(set(words)) / word_count if word_count else 0

    # Punctuation frequency
    puncts = ['.', ',', ';', ':', '!', '?', '"', "'", '-', '(', ')']
    punct_freqs = [text.count(p) / char_count for p in puncts]

    features.extend([
        avg_word_length,
        avg_sentence_length,
        lexical_diversity
    ])
    features.extend(punct_freqs)

    return features

def plot_pca(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    for label in np.unique(y):
        plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1],
                    label=f"Author {label + 1}", alpha=0.7, s=80)

    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('PCA Stylometric Cluster')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def plot_fingerprint(X_class):
    X_class = np.array(X_class)
    if len(X_class) > 1:
        avg_features = np.mean(X_class, axis=0)
    else:
        avg_features = X_class[0]

    num_points = len(avg_features)
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radius = avg_features / np.max(avg_features)

    x_outer = radius * np.cos(angles)
    y_outer = radius * np.sin(angles)

    plt.figure(figsize=(6, 6))
    plt.plot(np.append(x_outer, x_outer[0]), np.append(y_outer, y_outer[0]),
             c='darkblue', lw=2, alpha=0.8)

    for i, j in combinations(range(num_points), 2):
        dist = np.linalg.norm([x_outer[i] - x_outer[j], y_outer[i] - y_outer[j]])
        if dist < 1.5:  # Connect nearby points
            plt.plot([x_outer[i], x_outer[j]], [y_outer[i], y_outer[j]], c='skyblue', lw=0.5, alpha=0.5)

    plt.scatter(x_outer, y_outer, c='darkred', s=40)

    for i in range(num_points):
        plt.text(x_outer[i]*1.1, y_outer[i]*1.1, f"F{i+1}", fontsize=8, ha='center')

    plt.title('Stylometric Fingerprint')
    plt.axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf
