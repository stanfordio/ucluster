from os import fdopen, remove
from typing import List
import fasttext
import nltk
import tempfile
from numpy import ndarray
from loguru import logger
from pyclustering.cluster import optics
import hdbscan
from fasttext.FastText import _FastText as FastTextModel

def preprocess_text(text: str) -> str:
    return " ".join(nltk.word_tokenize(text.lower()))

def train_word_vectors(post_texts: List[str]) -> FastTextModel:
    logger.info("Training word vectors...")
    fd, path = tempfile.mkstemp()
    with fdopen(fd, "wb") as outfile:
        for text in post_texts:
            out = preprocess_text(text) + "\n"
            outfile.write(out.encode("utf8"))

    model = fasttext.train_unsupervised(path, model="skipgram")
    remove(path)
    logger.info("Word vectors trained!")
    return model

def _vectorize(text: str, model: FastTextModel) -> ndarray:
    return model.get_sentence_vector(preprocess_text(text))

def cluster_texts(texts: List[str], model: FastTextModel) -> ndarray:
    logger.info("Clustering texts...")
    vectors = [_vectorize(text, model) for text in texts]
    clusters = optics.optics(vectors, 0.05, 5, ccore=True)
    clusters.process()
    predictions = clusters.get_clusters()
    logger.info("Clustering complete!")
    return predictions

def display_clusters(texts: List[str], clusters: List[List[int]]):
    for i in range(len(clusters)):
        print(f"Cluster #{i+1} -----")
        for idx in clusters[i]:
            print(f"{idx}: {texts[idx]}")

if __name__ == "__main__":
    with open("data/buzz.txt", "r") as infile:
        lines = infile.readlines()
    model = train_word_vectors(lines)
    cs = cluster_texts(lines, model)
    display_clusters(lines, cs)