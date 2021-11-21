from os import fdopen, remove
from typing import List
import fasttext
import nltk
import tempfile
from numpy import ndarray
from loguru import logger
from collections import defaultdict
import hdbscan


def preprocess_text(text: str) -> str:
    return " ".join(nltk.word_tokenize(text.lower()))


class TextClusterer:
    def fit(self, texts: List[str]) -> None:
        raise NotImplemented

    def clusters(self) -> List[int]:
        raise NotImplemented

    def probabilities(self) -> List[float]:
        raise NotImplemented

    def outlier_probabilities(self) -> List[float]:
        raise NotImplemented


class FuzzyClusterer(TextClusterer):
    def __init__(self, dims=25, min_cluster_size=3, min_samples=3):
        self.dims = dims
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self._model = None

    def _train_word_vectors(
        self,
        texts: List[str],
    ) -> None:
        logger.info("Training word vectors...")
        fd, path = tempfile.mkstemp()
        with fdopen(fd, "wb") as outfile:
            for text in texts:
                out = preprocess_text(text) + "\n"
                outfile.write(out.encode("utf8"))

        model = fasttext.train_unsupervised(path, model="skipgram", dim=self.dims)
        remove(path)
        logger.info("Word vectors trained!")
        self._model = model

    def _vectorize(self, text: str) -> ndarray:
        return self.model.get_sentence_vector(preprocess_text(text))

    def _cluster_texts(self, texts: List[str]) -> None:
        logger.info("Clustering texts...")
        vectors = [self._vectorize(text, model) for text in texts]
        clusters = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size, min_samples=self.min_samples
        )
        self._predictions = clusters.fit_predict(vectors)
        logger.info("Clustering complete!")
        self._clusterer = clusters

    def fit(self, texts: List[str]) -> None:
        self._train_word_vectors(texts)
        self._cluster_texts()

    def clusters(self) -> List[int]:
        return self._predictions

    def probabilities(self) -> List[float]:
        return self._clusterer.probabilities_

    def outlier_probabilities(self) -> List[float]:
        return self._clusterer.outlier_scores_


class ExactClusterer:
    def fit(self, texts: List[str]) -> None:
        dupes = defaultdict(lambda: [])
        for idx, text in enumerate(texts):
            dupes[preprocess_text(text)].append(idx)

        clusters = {}
        i = 0
        for _, occurences in dupes.items():
            if len(occurences) <= 1:
                continue
            for occurence in occurences:
                clusters[occurence] = i
            i += 1

        self._clusters = [clusters.get(i, -1) for i in range(len(texts))]

    def clusters(self) -> List[int]:
        return self._clusters

    def probabilities(self) -> List[float]:
        return [1 if c != -1 else 0 for c in self.clusters()]

    def outlier_probabilities(self) -> List[float]:
        return [1 if c == -1 else 0 for c in self.clusters()]


def _display_clusters(
    texts: List[str], clusters: List[List[int]], probabilities: List[float]
):
    cluster_assignments = defaultdict(lambda: [])
    for i, assignment in enumerate(clusters):
        cluster_assignments[assignment].append(i)
    for cluster, indices in cluster_assignments.items():
        print(f"Cluster {cluster} -----")
        for idx in indices:
            print(f"{cluster} {probabilities[idx]}: {texts[idx].strip()}")


if __name__ == "__main__":
    with open("data/buzz_full.txt", "r") as infile:
        lines = infile.readlines()
    cl = ExactClusterer()
    cl.fit(lines)
    _display_clusters(lines, cl.clusters(), cl.probabilities())
