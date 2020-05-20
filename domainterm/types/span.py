#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Span:
    def __init__(self, sentence, start, end):
        self.sentence = sentence
        self.start = start
        self.end = end

    def __str__(self):
        return " ".join([str(t) for t in self.sentence.tokens[self.start:self.end]])

    def __add__(self, other):
        if self.sentence == other.sentence and self.end == other.start:
            return Span(self.sentence, self.start, other.end)
        return None
