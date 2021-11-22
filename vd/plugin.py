"""This plugin adds text and user clustering."""

import visidata as vd
from visidata import BaseSheet, Sheet, Column
from visidata.column import SettableColumn
from visidata.vdobj import asyncthread
from visidata.threads import Progress
import ucluster

__author__ = "R. Miles McCain <github@sendmiles.email>"
__version__ = "1.0.0"


@asyncthread
def cluster(col: Column, clusterer: ucluster.TextClusterer, col_name: str):
    sheet = col.sheet
    rows = sheet.rows

    clusters = SettableColumn(col.name + "_" + col_name + "_cluster")
    sheet.addColumn(clusters, index=sheet.columns.index(col) + 1)

    probs = SettableColumn(col.name + "_" + col_name + "_cluster_prob")
    sheet.addColumn(probs, index=sheet.columns.index(col) + 2)

    outliers = SettableColumn(col.name + "_" + col_name + "_cluster_outlier")
    sheet.addColumn(outliers, index=sheet.columns.index(col) + 3)

    texts = []
    for r in Progress(rows, gerund="reading values"):
        val = col.getValue(r)
        texts.append(str(val) if val is not None else "")

    clusterer.fit(texts)

    for r, v, p, o in zip(
        rows,
        clusterer.clusters(),
        clusterer.probabilities(),
        clusterer.outlier_probabilities(),
    ):
        if v != -1:
            clusters.setValue(r, v)
        probs.setValue(r, p)
        outliers.setValue(r, o)


@Column.api
def exact_cluster(col: Column):
    cluster(col, ucluster.ExactClusterer(), col_name="exact")


@Column.api
def fuzzy_cluster(col: Column):
    cluster(col, ucluster.FuzzyClusterer(), col_name="fuzzy")


BaseSheet.addCommand(None, "exact-cluster", "cursorCol.exact_cluster()")
BaseSheet.addCommand(None, "fuzzy-cluster", "cursorCol.fuzzy_cluster()")
