# Sentence Coherence Reordering

Python-based NLP toolkit to detect if sentences in a paragraph are logically ordered and automatically reorder shuffled sentences into a coherent flow.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Sentence--BERT-orange?style=flat" alt="Sentence-BERT">
  <img src="https://img.shields.io/badge/ROCStories Dataset-purple?style=flat" alt="Dataset">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat" alt="MIT">
</p>

**Goal**: Given raw or jumbled text, split it into sentences, score coherence of their order, and reconstruct the most logical paragraph — no numbered labels needed.

### ✨ Key Features

- Fully automatic sentence splitting with NLTK Punkt
- Semantic embeddings using **Sentence-BERT** (all-MiniLM-L6-v2)
- Pairwise sentence order prediction via lightweight neural network
- Global ranking to find the best overall coherent sequence
- Works on short-to-medium paragraphs (3–10 sentences)
- Trained on **ROCStories** dataset for natural narrative coherence
- CPU-friendly inference — no GPU required
- Simple one-command train + inference pipeline
- Modular design: easy to swap encoders or add features
