#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .span import Span

class Sentence:
    def __init__(self, text, tokens=None, nps=None, sources=None):
        self.text = text
        self.tokens = tokens
        self.nps = nps if nps else set()
        self.sources = sources if sources else set()

    def __str__(self):
        return self.text

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)

    def tokenized_text(self):
        return " ".join([token.text for token in self.tokens])

    def lemma_text(self):
        return " ".join([token.lemma for token in self.tokens])

    def emb(self):
        return sum([token.emb() for token in self.tokens]) / len(self)

    def add_nps(self, *nps):
        self.nps.update({Span(self, start, end) for start, end in nps})

    def find_spans(self, *spans, is_lemma=False, ignore_case=False):
        words = [token.lemma if is_lemma else token.text for token in self.tokens]
        words = [word.lower() if ignore_case else word for word in words]
        norm2span = {}
        group_dict = {}
        for span in spans:
            norm = span.lower() if ignore_case else span
            norm2span[norm] = span
            length = len(norm.split())
            if length in group_dict:
                group_dict[length].add(norm)
            else:
                group_dict[length] = {norm}
        group_dict = dict(sorted(group_dict.items(), key=lambda x: x[0], reverse=True))
        index = 0
        result_dict = {}
        while index < len(self):
            for length, group in group_dict.items():
                span = " ".join(words[index:index + length])
                if span in group:
                    if norm2span[span] not in result_dict:
                        result_dict[norm2span[span]] = set()
                    result_dict[norm2span[span]].add((index, index + length))
                    index += length
                    break
            else:
                index += 1
        # result = [(span, result_dict.get(span, -1), ) for span in spans]
        return result_dict