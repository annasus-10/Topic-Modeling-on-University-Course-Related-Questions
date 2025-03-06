# Topic Modeling for University Course Questions

## Overview

This repository contains a topic modeling pipeline designed to analyze university course-related questions. The project utilizes natural language processing (NLP) techniques to identify patterns and categorize questions into meaningful topics. The extracted topics help in understanding common themes in student queries.

The questions used in this project were extracted from previously generated text-to-SQL samples. Only the **natural language questions** were used as input for the topic modeling pipeline, ensuring that the dataset consists of real student-like queries.

This project was implemented and run in **Google Colab** for easy execution and visualization.

This is an ongoing project, and further refinements will be made to improve the accuracy and effectiveness of the topic modeling approach.

Repository for text-to-SQL sample generation: https\://github.com/annasus-10/Generate-Text-and-SQL-Pairs

## Features

- **Data Preprocessing**: Cleans and normalizes text data, including tokenization, stopword removal, and lemmatization.
- **TF-IDF Vectorization**: Converts text into numerical representations to highlight important terms.
- **Latent Dirichlet Allocation (LDA)**: Implements topic modeling to extract hidden themes from the questions dataset.
- **Dimensionality Reduction (t-SNE)**: Visualizes high-dimensional topic distributions in a 2D space.
- **Hierarchical Clustering**: Groups similar questions based on topic similarity.
- **Word Cloud Visualization**: Displays key terms from the dataset to identify frequently used words.

## Dependencies

- Python 3.x
- `pandas` (for data handling)
- `numpy` (for numerical computations)
- `spacy` (for NLP preprocessing)
- `nltk` (for stopword removal and lemmatization)
- `gensim` (for topic modeling with LDA)
- `scikit-learn` (for TF-IDF vectorization and clustering)
- `matplotlib` & `seaborn` (for visualization)
- `wordcloud` (for generating word clouds)

## Usage

### 1. Setup

Ensure you have the necessary dependencies installed:

```bash
pip install pandas numpy spacy nltk gensim scikit-learn matplotlib seaborn wordcloud
```

Additionally, download the required NLP models:

```bash
python -m spacy download en_core_web_sm
```

### 2. Running the Script

Execute the script to preprocess the questions and perform topic modeling:

```bash
python topic_modeling.py
```

### 3. Output

- **Topic Distribution**: A set of extracted topics with the most representative words.
- **t-SNE Plot**: A scatter plot showing clustered topics.
- **Dendrogram**: A hierarchical clustering visualization.
- **Word Cloud**: A graphical representation of frequently occurring terms.

## Future Improvements

- **Fine-Tuning LDA Parameters**: Optimize topic coherence and interpretability.
- **Alternative Topic Modeling Methods**: Experiment with **BERTopic** or **Non-Negative Matrix Factorization (NMF)**.
- **Interactive Visualization**: Implement dashboards for better topic exploration.

This repository is actively being developed, and improvements will continue to refine the methodology and results.

