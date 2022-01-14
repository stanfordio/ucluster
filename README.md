# uCluster
uCluster is a simple and fast way to cluster textual content. It is available both as a standalone Python package as well as a [VisiData](https://visidata.org) plugin. This documentation will focus on using uCluster as a VisiData plugin; to see how to use it as a Python package, simply inspect [`ucluster/text_cluster.py`](ucluster/text_cluster.py) — it's a very simple interface.

## Motivation

Suppose you have a dataset of millions of text posts. That's great! But there's no way you're going to be able to sift through each post one-by-one. Sure, you can do an emoji flag analysis to get a sense of national identities; similarly, you can look at n-grams to get a sense of popular phrases. These are both useful approaches, but they don't give you a great sense for *communities* of content. Are there a bunch of posts in the dataset that are essentially saying the same thing?

uCluster uses (relatively simple) NLP techniques to *cluster* datasets of text. It looks for similarities in the posts, and groups related posts — for some rough understanding of "related" — together. Not every post will be assigned to a cluster, and not every cluster will be a perfect grouping. As an exploratory tool, however, uCluster can be quite powerful. If a lot of posts are saying essentially the same thing — even with small variations in phrasing, punctuation, emojis, etc. — uCluster will notice.

## Interpreting Clusters

## Words of Warning

uCluster is not a magic tool, and it's important to be mindful of its inherent limitations.

1. **Clusters aren't significant by themselves.** The clusters exist only to point you towards potentially related content — there is nothing significant about the clusters themselves. For example, if two users' posts frequently cluster together, that is not a sign of coordinated inauthentic behavior; it's simply a sign that you might want to investigate those users (and those clusters) further.

2. **It handles English best (but it's still somewhat multilingual).** uCluster (and the tools it uses under the hood) make several assumptions about the input text, which tend to be most accurate for English text, usually accurate for latin-charactered space-separated languages (e.g., Portuguese), kind of accurate for other space-separated languages (e.g., Russian), and least accurate for character-based languages (e.g., Chinese). You can use uCluster for multilingual datasets (this is one of its key design goals), but don't be surprised if you end up with a cluster that is characterized not by content, per se, but by language (e.g., a cluster that contains all the Arabic posts in your dataset).

3. **It uses a lot of memory.** uCluster doesn't use pre-trained word vectors; this is what makes it possible to cluster multilingual content, as well as what allows it to perform so well on social media content. Rather, it trains its word vectors from scratch each time you run it. This means that it uses quite a bit of memory. You might want to run this on a powerful server somewhere — not on your laptop.

4. **It assumes posts are relatively short.** uCluster works best on posts that are relatively short (think tweet-length). That makes it great for datasets from platforms like Twitter, Gab, Gettr, and Parler, where posts tend to be only a few sentences. It's going to be much less effective for classifying, say, Medium posts.

5. **It doesn't perform any meaningful pre-processing.** uCluster doesn't strip HTML tags, links, or @mentions, for example. Be sure to clean your dataset to make it as close to "pure" text content as possible before running it through uCluster. (Don't strip emojis, though — anecdotally, these are very helpful for clustering. They carry a lot of meaning!)

## VisiData Usage

## In The Weeds: Architecture & Design