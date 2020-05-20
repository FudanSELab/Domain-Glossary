#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .vocab import Vocab


class Token:
    def __init__(self, vocab:Vocab, text, lemma, pos=None, dep=None, ner=None):
        self.vocab = vocab
        self.text = text
        self.lemma = lemma
        self.lemma_first = vocab.lemma_first
        self.vocab.add_word(lemma if self.lemma_first else text)
        self.pos = pos
        self.dep = dep
        self.ner = ner

    def __str__(self):
        return self.lemma if self.lemma_first else self.text

    def __hash__(self):
        return hash(str(self))

    def emb(self):
        return self.vocab.get_emb(str(self))
