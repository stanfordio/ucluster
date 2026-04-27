"""This plugin adds text clustering commands to VisiData."""

import threading

from visidata.column import Column, SettableColumn
from visidata.sheets import BaseSheet
from visidata.threads import Progress
from visidata.vdobj import asyncthread

import ucluster

__author__ = "R. Miles McCain <github@sendmiles.email>"
__version__ = "2.0.0"


@asyncthread
def cluster(col: Column, clusterer: ucluster.TextClusterer, col_name: str) -> None:
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

    for r, v in zip(rows, clusterer.clusters(), strict=True):
        clusters.setValue(r, v)


@Column.api
def exact_cluster(col: Column) -> None:
    cluster(col, ucluster.ExactClusterer(), col_name="exact")


@Column.api
def fuzzy_cluster(col: Column) -> None:
    cluster(col, ucluster.FuzzyClusterer(), col_name="fuzzy")


BaseSheet.addCommand(None, "exact-cluster", "cursorCol.exact_cluster()")
BaseSheet.addCommand(None, "fuzzy-cluster", "cursorCol.fuzzy_cluster()")
