#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import re

import spacy
from gensim.models import Word2Vec

from .vocab import Vocab
from .token import Token
from .sentence import Sentence

class Corpus:
    def __init__(self,
            vocab=Vocab(),
            sentences=None,
            code_elements=None,
            code_relations=None,
            seeds=None,
            terms=None,
            concepts=None,
            relations=None,

            general_vocab=Vocab(),
            general_sentences=None,
        ):
        self.vocab = vocab
        self.sentences = sentences if sentences else set()
        self.code_elements = code_elements if code_elements else set()
        self.code_relations = code_relations if code_relations else set()
        self.seeds = seeds if seeds else set()
        self.terms = terms if terms else set()
        self.concepts = concepts if concepts else set()
        self.relations = relations if relations else set()

        self.general_vocab = general_vocab
        self.general_sentences = general_sentences if general_sentences else set()
        

   