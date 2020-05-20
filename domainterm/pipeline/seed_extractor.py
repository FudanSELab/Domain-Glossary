#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm

from .pipe import Pipe
from .term_validater import TermValidater
from ..types import Corpus
from ..seed_rules import Rule, RULES
from ..option import Option


class SeedExtractor(Pipe):
    def __init__(self, **cfg):
        super(SeedExtractor, self).__init__(**cfg)
        self.rules = [RULE() for RULE in RULES]
        self.term_validater = TermValidater.load(Option.GENERAL_CORPUS)

    def process(self, corpus:Corpus):
        seeds = set()
        for sent in tqdm(corpus.sentences, desc=' - (Seed Extractor)', leave=False, ascii=True):
            ## Pattern
            for rule in self.rules:
                terms, _ = rule.extract(sent.text)
                seeds.update(terms)
            ## Code and Text
            matching_result = sent.find_spans(*corpus.code_elements, ignore_case=True)
            terms = {term for term, indices in matching_result.items() if len(indices) > 0}
            seeds.update(terms)
        print("seeds before validation:", len(seeds))
        seeds = self.term_validater.validate_lexicon(seeds, stopword_level=0)
        seeds = self.term_validater.validate_origin(seeds, corpus.sentences)
        seeds = self.term_validater.validate_freq(seeds, corpus.sentences, corpus.vocab)
        print("seeds after validation:", len(seeds))
        corpus.seeds = seeds
        return corpus
        