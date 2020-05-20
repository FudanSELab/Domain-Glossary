#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import numpy as np


class Vocab:
    def __init__(self, lemma_first=False, stopwords=None, emb_size=300):
        self.words = set()
        self.embeddings = dict()
        self.lemma_first = lemma_first
        self.stopwords = stopwords if stopwords else set()
        self.emb_size = emb_size
        self.ZERO = np.array([0.] * self.emb_size)

    def get_emb(self, word):
        return self.embeddings.get(word, self.ZERO)

    def add_word(self, word):
        self.words.add(word)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, key):
        return self.get_emb(key)

    def __add__(self, other):
        vocab = Vocab(self.lemma_first, self.stopwords, self.emb_size)
        vocab.words = self.words | other.words
        vocab.embeddings = dict(set(self.embeddings.items()) | (other.embeddings.items()))
        vocab.stopwords = self.stopwords
        if self.stopwords is not None:
            if other.stopwords is not None:
                self.stopwords.update(other.stopwords)
        else:
            self.stopwords = other.stopwords
        return vocab

    def __iadd__(self, other):
        self.words |= other.words
        self.embeddings.update(other.embeddings)
        if self.stopwords is not None:
            if other.stopwords is not None:
                self.stopwords.update(other.stopwords)
        else:
            self.stopwords = other.stopwords
        return self

    # @classmethod
    # def load(file_name):
        
