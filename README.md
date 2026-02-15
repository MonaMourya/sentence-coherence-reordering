Sentence Order Coherence Detection and Reordering

An NLP project that detects whether the sentences in a paragraph are logically ordered and automatically reorders them to restore coherence using semantic sentence embeddings and machine learning.

📌 Overview

Sentence coherence refers to the logical flow of ideas across sentences in a paragraph. Even when individual sentences are grammatically correct, an incorrect ordering can make the text confusing or unreadable.

This project:

Takes raw, unstructured text as input

Detects whether the sentence order is coherent or incoherent

Reorders sentences to produce a logically coherent paragraph

The system does not require sentence labels (S1, S2, etc.) and works directly on natural text blocks.

✨ Features

Automatic sentence segmentation from raw text

Semantic sentence representations using Sentence-BERT

Pairwise sentence order prediction

Robust global sentence reordering

CPU-friendly (no GPU required)

Modular and extensible architecture

🏗️ Project Architecture
sentence-coherence-reordering/
│
├── models/
│   ├── sentence_encoder.py
│   ├── coherence_model.py
│   └── trained/
│
├── training/
│   ├── dataset_builder.py
│   └── train.py
│
├── inference/
│   ├── coherence_check.py
│   └── reorder.py
│
├── utils/
│   ├── sentence_splitter.py
│   └── metrics.py
│
├── input/
│   └── sample.txt
│
├── main.py
├── requirements.txt
└── README.md
🧠 Algorithms Used
1. NLTK Punkt Sentence Tokenizer

Automatically splits raw text into sentences

Requires no manual formatting or sentence labels

2. Sentence-BERT (SBERT)

Converts each sentence into a dense semantic embedding

Captures contextual meaning beyond word-level features

Model used: all-MiniLM-L6-v2

3. Pairwise Sentence Order Classification

A lightweight neural network predicts whether one sentence should come before another

Learns logical relations such as:

cause → effect

goal → action → outcome

temporal progression

4. Global Score-Based Sentence Ranking

Aggregates pairwise predictions into a global score for each sentence

Produces a stable sentence order

Avoids cyclic dependencies common in graph-based approaches

📊 Dataset Used for Training

ROCStories dataset (via Hugging Face)

Contains thousands of short, coherent English stories

Each story consists of multiple logically ordered sentences

Suitable for learning sentence-level coherence patterns

▶️ How to Run
1️⃣ Install Dependencies
pip install -r requirements.txt
2️⃣ Train the Model (One-Time)
python -m training.train

This step:

Loads the ROCStories dataset

Trains the coherence model

Saves trained weights to models/trained/

3️⃣ Run Inference

Edit input/sample.txt with any paragraph (ordered or unordered), then run:

python main.py
🧪 Example

Input (Unordered):

He won the race.
He trained every day.
He dreamed of becoming a runner.

Output (Reordered):

He dreamed of becoming a runner.
He trained every day.
He won the race.

⚠️ Limitations

Best performance on short to medium paragraphs (3–8 sentences)

Performance degrades for very long texts due to quadratic pairwise comparisons

Some texts may have multiple valid sentence orders

Designed for English (can be extended to other languages)

🚀 Future Work

Hierarchical coherence modeling for long documents

Discourse-aware and temporal reasoning

Multilingual sentence coherence detection

Integration with text generation systems

Evaluation using metrics such as Kendall’s Tau

🎓 Academic Use

This project is suitable for:

Final-year NLP projects

Research prototypes

Demonstrations of sentence-level discourse modeling