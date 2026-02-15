import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")

def split_sentences(text):
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]