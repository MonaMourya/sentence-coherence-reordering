import torch

from utils.sentence_splitter import split_sentences
from models.sentence_encoder import SentenceEncoder
from models.coherence_model import CoherenceModel
from inference.coherence_check import compute_probability_matrix
from inference.reorder import reorder_sentences
from utils.metrics import is_coherent

# LOAD INPUT
with open("input/sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

sentences = split_sentences(text)

print("\n🔹 Original Sentences:")
for s in sentences:
    print("-", s)

# LOAD MODELS
encoder = SentenceEncoder()
model = CoherenceModel()
model.load_state_dict(torch.load("models/trained/coherence_model.pt"))

# INFERENCE
prob_matrix = compute_probability_matrix(sentences, encoder, model)
order = reorder_sentences(prob_matrix)

print("\n🔹 Reordered Paragraph:\n")
for idx in order:
    print(sentences[idx])

print("\n🔹 Coherence Check:", "Coherent" if is_coherent(order) else "Incoherent")