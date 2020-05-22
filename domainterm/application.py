#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from pathlib import Path
import pickle

from .pipeline import *
from .option import Option

PIPE_FACTORY = {
    "preprocessor": Preprocessor,
    "seed_extractor": SeedExtractor,
    "term_recognizer": TermRecognizer,
    "alias_fusion": AliasFusion,
    "relation_extractor": RelationExtrator,
}

DEFAULT_PIPELINE = [
    "preprocessor",
    "seed_extractor",
    "term_recognizer",
    "alias_fusion",
    "relation_extractor"
]

class App:
    def __init__(self, pipeline=DEFAULT_PIPELINE):
        self.pipeline = [PIPE_FACTORY[pipe]().with_name(pipe) for pipe in pipeline]

    def __call__(self, *data):
        self.process(*data)

    def process(self, *data, output=None):
        corpus = None
        for pipe in self.pipeline:
            print("{} start ({}).".format(pipe.name, time.asctime(time.localtime(time.time()))))
            corpus = pipe(*data)
            print("{} finish ({}).".format(pipe.name, time.asctime(time.localtime(time.time()))))
            if output:
                with (Path(output) / f"{pipe.name}.corpus").open("wb") as f:
                    pickle.dump(corpus, f)
            data = (corpus, )
        return corpus
