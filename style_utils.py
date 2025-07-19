import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def extract_combined_features(text):
    text = text.lower()
    words = text.split()
    num_words = len(words)
    num_sentences = text.count('.') + text.count('!') + text.count('?')
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    vocab_richness = len(set(words)) / num_words if num_words else 0
    punctuation_freq = sum(text.count(p) for p in ',;:') / len(text) if text else 0
    return [num_words, num_sentences, avg_word_len, vocab_richness, punctuation_freq]

def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if len(text[i:i+chunk_size]) > 100]

def plot_pca(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig, ax = plt.subplots()
    ax.scatter(X_pca[y==0, 0], X_pca[y==0, 1], label="Author 1", alpha=0.6)
    ax.scatter(X_pca[y==1, 0], X_pca[y==1, 1], label="Author 2", alpha=0.6)
    ax.legend()
    ax.set_title("PCA of Author Features")
    return fig

def get_fingerprint_plot(X, y, label):
    X_label = X[y == label]
    fingerprint = get_author_fingerprint(X_label)
    fig, ax = plt.subplots()
    ax.bar(range(len(fingerprint)), fingerprint)
    ax.set_title(f"Fingerprint of Author {label + 1}")
    return fig

def get_author_fingerprint(vectors):
    return np.mean(vectors, axis=0)
