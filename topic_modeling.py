# -*- coding: utf-8 -*-
"""TextMining_Colab.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-rqcxBbdux3fGmKvFk4xh2v590CDzNJT
"""

import os
import json
import pandas as pd
import numpy as np
import spacy
import gensim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram
from wordcloud import WordCloud
from gensim.models import Word2Vec

# Download necessary NLP resources
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load JSON Data
file_path = "questions_only.json"  # Ensure the file path is correct
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File {file_path} not found. Please check the path.")

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data, columns=["text"])

# Text Preprocessing using spaCy
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    doc = nlp(text.lower())  # Use spaCy for tokenization
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(preprocess_text)

# Vectorization: TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df["clean_text"])

# Word Cloud Visualization
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(df["clean_text"]))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Processed Text")
plt.show()

# Topic Modeling using Latent Dirichlet Allocation (LDA)
num_topics = 5
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_topics = lda.fit_transform(tfidf_matrix)

# Displaying LDA Topics
feature_names = tfidf_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx+1}: ", [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]])

# Dimensionality Reduction using t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results = tsne.fit_transform(lda_topics)
df["tsne_x"], df["tsne_y"] = tsne_results[:, 0], tsne_results[:, 1]

# Scatter Plot of t-SNE Clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(x="tsne_x", y="tsne_y", hue=df.index % num_topics, palette="viridis", data=df)
plt.title("t-SNE Visualization of LDA Topics")
plt.show()

# Hierarchical Clustering
linked = linkage(tsne_results, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

# Box Plot of Word Count Distribution
df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["word_count"])
plt.title("Distribution of Word Count in Questions")
plt.xlabel("Word Count")
plt.show()

# Save Preprocessed Data
output_file = "preprocessed_text.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ Process Complete! Preprocessed data saved as '{output_file}'")