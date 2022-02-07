from functools import lru_cache
from os import fdopen, remove
from typing import List
import fasttext
import nltk
import tempfile
from numpy import ndarray
from loguru import logger
from collections import defaultdict
import hdbscan
import json

# We want logging disabled whenever we are running as a library; we only want it enabled for
# debugging purposes, so we just enable it explicitly when __name__ == "__main__".
logger.disable(__name__)

# Ensure it's present. Won't redownload.
nltk.download("punkt")


def preprocess_text(text: str) -> str:
    text = text.encode("utf-8", "replace").decode()  # Make everything play nice
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


class TransformerCluster(TextClusterer):
    def __init__(
        self, min_cluster_size=3, min_samples=3, alpha=1.0, epsilon=0.0
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.alpha = alpha
        self.epsilon = epsilon

    def _cluster_texts(self, texts: List[str]) -> None:
        logger.info("Encoding texts...")
        vectors = self.model.encode(texts)
        logger.info("Clustering texts...")
        clusters = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            alpha=self.alpha,
            cluster_selection_epsilon=self.epsilon,
        )
        self._predictions = clusters.fit_predict(vectors)
        logger.info("Clustering complete!")
        self._clusterer = clusters

    def fit(self, texts: List[str]) -> None:
        self._cluster_texts(texts)

    def clusters(self) -> List[int]:
        return self._predictions

    def probabilities(self) -> List[float]:
        return self._clusterer.probabilities_

    def outlier_probabilities(self) -> List[float]:
        return self._clusterer.outlier_scores_

class FuzzyClusterer(TextClusterer):
    def __init__(
        self, dims=25, min_cluster_size=3, min_samples=3, alpha=1.0, epsilon=0.0
    ):
        self.dims = dims
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.alpha = alpha
        self.epsilon = epsilon
        self._model = None

    def _train_word_vectors(
        self,
        texts: List[str],
    ) -> None:
        logger.info("Training word vectors...")
        fd, path = tempfile.mkstemp()
        logger.info(f"Using tempfile: {path}")
        with fdopen(fd, "wb") as outfile:
            for text in texts:
                out = preprocess_text(text) + "\n"
                outfile.write(out.encode("utf-8", "replace"))

        model = fasttext.train_unsupervised(
            path, model="skipgram", dim=self.dims, verbose=0
        )
        remove(path)
        logger.info("Word vectors trained!")
        self._model = model

    def _vectorize(self, text: str) -> ndarray:
        return self._model.get_sentence_vector(preprocess_text(text))

    def _cluster_texts(self, texts: List[str]) -> None:
        logger.info("Clustering texts...")
        vectors = [self._vectorize(text) for text in texts]
        clusters = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            alpha=self.alpha,
            cluster_selection_epsilon=self.epsilon,
        )
        self._predictions = clusters.fit_predict(vectors)
        logger.info("Clustering complete!")
        self._clusterer = clusters

    def fit(self, texts: List[str]) -> None:
        self._train_word_vectors(texts)
        self._cluster_texts(texts)

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
    logger.enable(__name__)

    with open("data/gettr_posts_small.jsonl", "r") as infile:
        posts = [json.loads(l) for l in infile.readlines()]

    logger.info("Getting text data from input...")
    text_data = [post.get("txt") or "" for post in posts]

    logger.info("Clustering...")
    cl = TransformerCluster()
    cl.fit(text_data)

    logger.info("Writing to file...")
    with open("data/clustered.jsonl", "w") as outfile:
        for post, cluster in zip(posts, cl.clusters()):
            post["_cluster"] = str(cluster)
            print(json.dumps(post), file=outfile)

    logger.info("Done writing clusters to the file!")
