# uCluster
uCluster is a simple and fast way to cluster textual content. It is available both as a standalone Python package as well as a [VisiData](https://visidata.org) plugin. This documentation will focus on using uCluster as a VisiData plugin; to see how to use it as a Python package, simply inspect [`ucluster/text_cluster.py`](ucluster/text_cluster.py) — it's a very simple interface.

## Motivation

Suppose you have a dataset of millions of text posts. That's great! But there's no way you're going to be able to sift through each post one-by-one. Sure, you can do an emoji flag analysis to get a sense of national identities; similarly, you can look at n-grams to get a sense of popular phrases. These are both useful approaches, but they don't give you a great sense for *communities* of content. Are there a bunch of posts in the dataset that are essentially saying the same thing?

uCluster uses (relatively simple) NLP techniques to *cluster* datasets of text. It looks for similarities in the posts, and groups related posts — for some rough understanding of "related" — together. Not every post will be assigned to a cluster, and not every cluster will be a perfect grouping. As an exploratory tool, however, uCluster can be quite powerful. If a lot of posts are saying essentially the same thing — even with small variations in phrasing, punctuation, emojis, etc. — uCluster will notice.

## Interpreting Clusters

When you run uCluster on your text dataset, each piece of text will either be assigned a cluster (identified by a number >= 0), no cluster (identified by the number -1). You can think of each cluster as containing "similar" posts — whether that similarity arises from using one particularly rare word (or hashtag), from using a similar sentence structure, or from something else. You can probably get an intuition for "why" each cluster is as it is from just looking at its content.

There is no pre-set number of clusters, as clusters are determined "naturally" by looking at density. For more information about how this clustering works, see the "architecture" section below.

**Note:** uCluster also supports _exact clustering_, in which it doesn't perform any NLP magic and simply creates clusters based on exact matches. For example, if there are multiple posts with the exact same content (case insensitive), they will be assigned to the same cluster. You can access this clusterer through the `exact-cluster` VisiData command, or through the `ExactClusterer` class in [`ucluster/text_cluster.py`](ucluster/text_cluster.py). Most of this document is referring to the more advanced "fuzzy" clusterer. 

## Words of Warning

uCluster is not a magic tool, and it's important to be mindful of its inherent limitations.

1. **Clusters aren't significant by themselves.** The clusters exist only to point you towards potentially related content — there is nothing significant about the clusters themselves. For example, if two users' posts frequently cluster together, that is not a sign of coordinated inauthentic behavior; it's simply a sign that you might want to investigate those users (and those clusters) further.

2. **It handles English best (but it's still somewhat multilingual).** uCluster (and the tools it uses under the hood) make several assumptions about the input text, which tend to be most accurate for English text, usually accurate for latin-charactered space-separated languages (e.g., Portuguese), kind of accurate for other space-separated languages (e.g., Russian), and least accurate for character-based languages (e.g., Chinese). You can use uCluster for multilingual datasets (this is one of its key design goals), but don't be surprised if you end up with a cluster that is characterized not by content, per se, but by language (e.g., a cluster that contains all the Arabic posts in your dataset).

3. **It uses a lot of memory.** uCluster doesn't use pre-trained word vectors; this is what makes it possible to cluster multilingual content, as well as what allows it to perform so well on social media content. Rather, it trains its word vectors from scratch each time you run it. This means that it uses quite a bit of memory. You might want to run this on a powerful server somewhere — not on your laptop.

4. **It assumes posts are relatively short.** uCluster works best on posts that are relatively short (think tweet-length). That makes it great for datasets from platforms like Twitter, Gab, Gettr, and Parler, where posts tend to be only a few sentences. It's going to be much less effective for classifying, say, Medium posts.

5. **It doesn't perform any meaningful pre-processing.** uCluster doesn't strip HTML tags, links, or @mentions, for example. Be sure to clean your dataset to make it as close to "pure" text content as possible before running it through uCluster. (Don't strip emojis, though — anecdotally, these are very helpful for clustering. They carry a lot of meaning!)

## VisiData Usage

When installed in VisiData, the uCluster plugin adds two commands that operate on columns: `fuzzy-cluster` and `exact-cluster`. Simply select the column that contains the text you would like to cluster, press space, then type `fuzzy-cluster` or `exact-cluster`. uCluster will then create a new column that will contain the cluster IDs. This can take several minutes. (Don't worry — it's done asynchronously, though VisiData still seems to occasionally freeze on giant datasets.)

## Installation

uCluster has several dependencies that are not pure Python (e.g., Facebook's [fastText](https://fasttext.cc/)), which means that it can be a bit of a pain to install.

The first step is to clone uCluster onto your local machine (`git clone git@github.com:stanfordio/ucluster.git`). Then follow the steps below.

#### 1. Install dependencies.

If you're on an x86 machine, everything should go relatively smoothly. With [Poetry](https://python-poetry.org/) installed, simply run `poetry install`. If you want to use uCluster in your "main" VisiData environment (i.e., you don't want to have to activate the Poetry virtual environment every time), then run `poetry env use $(which python3)` before running `poetry install`.

If you're on an M1 machine, things are a bit more complex. It doesn't seem possible to install SciPy (required by NLTK) using pip3 on the M1's, as it requires building SciPy from scratch. As a result, you'll need to use [Conda](https://conda.io). In the main project folder, run `conda env create -f env.yml`, then `conda activate ucluster`. Next, run `poetry env use $(which python3)` followed by `poetry install`.

uCluster will now be available in your Python environment.

#### 2. Install uCluster in VisiData

The next step is to install uCluster in VisiData. The approach described here is a bit hacky, so if you find a better way to do this, please submit a PR.

1. Make your local VisiData plugins directory if it doesn't yet exist with `mkdir -p ~/.visidata/plugins`.
2. Create a **hard link** from `~/.visidata/plugins/ucluster.py` to `vd/plugin.py` in the directory where you've locally cloned uCluster. You can do this by running `ln vd/plugin.py ~/.visidata/plugins/ucluster.py` from the main uCluster directory.
3. Tell VisiData about the plugin by adding the following line to your `~/.visidatarc` file: `import plugins.ucluster`.

You should now be able to use uCluster inside VisiData.

## In The Weeds: Architecture & Design

This section will eventually contain a more detailed explanation for how uCluster works. For now, here's a brief overview.

* First, we throw all the text into a giant file and use it to train word vectors using FastText.
* Next, we use those word vectors to turn each post into a vector. By default, we use 100-dimensional space.
* Finally, we run [HDBSCAN](https://hdbscan.readthedocs.io/) on those post vectors to create the final clusters. HDBSCAN is a high-performance general purpose clustering algorithm.

## Maintenance

uCluster is intended for internal SIO use, and currently maintained by Miles McCain. If you run into issues, ping the tech team and we will help.
