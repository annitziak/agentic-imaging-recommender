
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from gensim import corpora
from gensim.models.ldamodel import LdaModel

# This is the script used to evaluate retrieval strategies for scientific literature.
# The retrieval strategies (e.g. different number of queries, content of queries and number of retrieved evidence)
# are evaluate using 1. embedding similarity with the gold standard abstracts (their results sections)
# 2. using LDA to ensure thematic coverage of the same key concepts.

def preprocess_abstract(text):
    """Clean and structure the abstract text for processing. Extracts key sections and removes noise."""
    if isinstance(text, list):
        text = " ".join(text).strip()
    if not isinstance(text, str) or "not available" in text.lower():
        return ""
    text = text.strip()
    if not text:
        return ""
    
    # See if the abstract given is structured in sections like "Results", "Conclusions", etc.
    # If so, extract those sections
    structured = re.findall(
        r"(results?|conclusions?)[:\.]?\s*(.*?)(?=\n?[A-Z][a-z]+:|$)",
        text, flags=re.IGNORECASE | re.DOTALL
    )
    if structured:
        return " ".join([section.strip() for _, section in structured if section.strip()])

    # if no structured sections found, return last 3 sentences : we assume they are the most relevant to the "results"
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sentences[-3:]) if len(sentences) > 3 else text

def embed_abstracts(abstracts, model):
    """Embed a list of abstracts (their results part) using the provided SentenceTransformer model."""
    processed =  []
    for ab in abstracts:
        cleaned = preprocess_abstract(ab)
        if cleaned:
            processed.append(cleaned)
    embeddings = model.encode(processed, show_progress_bar=True)
    return embeddings, processed

def cluster_and_get_distribution(embeddings, n_clusters=5):
    """Cluster embeddings and return cluster labels, distribution, and centers."""
    if len(embeddings) < n_clusters:
        print(f"Too few samples to cluster. Try {n_clusters} or fewer.")
        return None, None, None
    
    # Use Kmeans clustering to make clusters in the embedding space
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    counts = np.bincount(labels, minlength=n_clusters)
    distribution = counts / counts.sum()
    return labels, distribution, kmeans.cluster_centers_

def average_min_center_distance(centers_a, centers_b, metric='euclidean'):
    """Calculate the average minimum distance between cluster centers of two sets."""
    dist_matrix = cdist(centers_a, centers_b, metric=metric)
    return (np.mean(np.min(dist_matrix, axis=1)) + np.mean(np.min(dist_matrix, axis=0))) / 2

def plot_embeddings_with_clusters(gold_embeds, retrieved_embeds, n_clusters=5, out_path="embedding_plot.png"):
    """Plot 2D PCA projection of embeddings with cluster centers for visualization."""
    all_embeds = np.vstack([gold_embeds, retrieved_embeds])

    # perform PCA
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_embeds)
    gold_2d = all_2d[:len(gold_embeds)] # for gold embeddings
    retrieved_2d = all_2d[len(gold_embeds):] # for retrieved embeddings

    kmeans_gold = KMeans(n_clusters=n_clusters, random_state=42).fit(gold_embeds)
    centers_gold_2d = pca.transform(kmeans_gold.cluster_centers_)
    kmeans_ret = KMeans(n_clusters=n_clusters, random_state=42).fit(retrieved_embeds)
    centers_ret_2d = pca.transform(kmeans_ret.cluster_centers_)

    # Plot the embeddings and cluster centers
    plt.figure(figsize=(8, 6))
    plt.scatter(gold_2d[:, 0], gold_2d[:, 1], c='#1f77b4', label='Gold abstracts', alpha=0.6, s=30)
    plt.scatter(retrieved_2d[:, 0], retrieved_2d[:, 1], c='#d62728', label='Retrieved abstracts', alpha=0.5, s=30)
    plt.scatter(centers_gold_2d[:, 0], centers_gold_2d[:, 1], c='#1f77b4', edgecolors='black', marker='X', s=220)
    plt.scatter(centers_ret_2d[:, 0], centers_ret_2d[:, 1], c='#d62728', edgecolors='black', marker='P', s=220)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Gold abstracts',
               markerfacecolor='#1f77b4', markersize=6, alpha=0.6),
        Line2D([0], [0], marker='o', color='w', label='Retrieved abstracts',
               markerfacecolor='#d62728', markersize=6, alpha=0.5),
        Line2D([0], [0], marker='X', color='black', label='Gold cluster centers',
               markerfacecolor='#1f77b4', markersize=10),
        Line2D([0], [0], marker='P', color='black', label='Retrieved cluster centers',
               markerfacecolor='#d62728', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks([]), plt.yticks([])
    plt.title("Semantic Embedding Space of Gold vs Retrieved Abstracts")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"📊 Saved plot to {out_path}")
    plt.show()


# Perform also LDA topic modeling on the abstracts to see overlap and topic distributions
def preprocess_tokens(text):
    """Tokenize and clean text for LDA topic modeling. Removes stop words and non-alphabetic characters."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()

    # very simple stop words list to avoid using them in LDA
    stop_words = {
        "the", "and", "of", "to", "in", "a", "for", "is", "on", "with", "that",
        "as", "are", "was", "by", "an", "be", "this", "which", "or", "at", "from",
        "it", "we", "has", "have", "not", "but", "were", "can", "also", "our"
    }
    return [t for t in tokens if t not in stop_words and len(t) > 2]


def run_lda(texts):
    """Run LDA topic modeling on a list of texts and print the topics."""
    tokenized = [preprocess_tokens(doc) for doc in texts]
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(text) for text in tokenized]
    lda = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10, random_state=42)
    for i, topic in lda.print_topics(num_words=10):
        print(f"Topic {i}: {topic}")



def main():
    # Load gold and retrieved abstracts from JSON files, they should both be in two json files with "abstract" keys
    gold_file = ""
    retrieved_file = ""

    with open(gold_file) as f:
        gold_data = json.load(f)
    with open(retrieved_file) as f:
        ret_data = json.load(f)

    gold_texts = [p.get("abstract", "") for p in gold_data]
    ret_texts = [p.get("abstract", "") for p in ret_data]

    model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    print("Model loaded.")

    # Embed and preprocess abstracts
    gold_embeds, gold_clean = embed_abstracts(gold_texts, model)
    ret_embeds, ret_clean = embed_abstracts(ret_texts, model)

    # Cluster and analyze distributions
    _, gold_dist, gold_centers = cluster_and_get_distribution(gold_embeds)
    _, ret_dist, ret_centers = cluster_and_get_distribution(ret_embeds)

    if gold_dist is not None and ret_dist is not None:
        print(f"Gold cluster distribution: {gold_dist}")
        print(f"Retrieved cluster distribution: {ret_dist}")
        print(f"Center Distance: {average_min_center_distance(gold_centers, ret_centers):.4f}")
        plot_embeddings_with_clusters(gold_embeds, ret_embeds)
    else:
        print("Skipping comparison due to insufficient clusters.")

    print("\n LDA on Retrieved Abstracts:")
    run_lda(ret_clean)
    print("\n LDA on Gold Abstracts:")
    run_lda(gold_clean)

if __name__ == "__main__":
    main()
