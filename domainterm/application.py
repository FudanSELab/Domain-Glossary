#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

from .pipeline import *
from .option import Option
from .util.saver import Saver

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

    def process(self, *data):
        corpus = None
        for pipe in self.pipeline:
            print("{} start ({}).".format(pipe.name, time.asctime(time.localtime(time.time()))))
            corpus = pipe(*data)
            print("{} finish ({}).".format(pipe.name, time.asctime(time.localtime(time.time()))))
            data = (corpus, )
        return corpus
