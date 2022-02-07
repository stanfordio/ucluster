"""This plugin adds text and user clustering!"""

import visidata as vd
from visidata import BaseSheet, Sheet, Column
from visidata import threads
from visidata.column import SettableColumn
from visidata.vdobj import asyncthread
from visidata.threads import Progress
import threading
import ucluster

__author__ = "R. Miles McCain <github@sendmiles.email>"
__version__ = "1.0.0"


@asyncthread
def cluster(col: Column, clusterer: ucluster.TextClusterer, col_name: str):
    sheet = col.sheet
    rows = sheet.rows

    clusters = SettableColumn(col.name + "_" + col_name + "_cluster")
    sheet.addColumn(clusters, index=sheet.columns.index(col) + 1)

    texts = []
    thread = threading.current_thread()
    for r in Progress(rows, gerund="reading values"):
        clusters.setValue(r, thread)
        val = col.getValue(r)
        texts.append(str(val) if val is not None else "")

    clusterer.fit(texts)

    for (
        r,
        v,
    ) in zip(rows, clusterer.clusters()):
        clusters.setValue(r, v)


@Column.api
def exact_cluster(col: Column):
    cluster(col, ucluster.ExactClusterer(), col_name="exact")


@Column.api
def fuzzy_cluster(col: Column):
    cluster(col, ucluster.FuzzyClusterer(), col_name="fuzzy")

@Column.api
def tf_cluster(col: Column):
    cluster(col, ucluster.TransformerCluster(), col_name="tf")


BaseSheet.addCommand(None, "exact-cluster", "cursorCol.exact_cluster()")
BaseSheet.addCommand(None, "fuzzy-cluster", "cursorCol.fuzzy_cluster()")
BaseSheet.addCommand(None, "tf-cluster", "cursorCol.tf_cluster()")
