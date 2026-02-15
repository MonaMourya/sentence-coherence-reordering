import random
from utils.sentence_splitter import split_sentences

def build_dataset(text, max_sentences=5):
    sentences = split_sentences(text)

    if len(sentences) < 3:
        return None

    sentences = sentences[:max_sentences]

    coherent = sentences
    incoherent = sentences.copy()
    random.shuffle(incoherent)

    return coherent, incoherent