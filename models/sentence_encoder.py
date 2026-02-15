from sentence_transformers import SentenceTransformer

class SentenceEncoder:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def encode(self, sentences):
        return self.model.encode(sentences, convert_to_tensor=True)