#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Concept:
    def __init__(self, aliases):
        self.aliases = aliases
        self.alias2count = dict()

    def name(self):
        count2aliases = dict()
        for alias in self.aliases:
            count = self.alias2count.get(alias, 0)
            if count not in count2aliases:
                count2aliases[count] = set()
            count2aliases[count].add(alias)

        _, candidates = max(count2aliases.items(), key=lambda item: item[0])
        return max(candidates, key=lambda x: len(x))

    def init_count(self, alias_count):
        for alias in self.aliases:
            self.alias2count[alias] = alias_count.get(alias, 0)

    def __iadd__(self, other):
        self.aliases |= other.aliases
        self.alias2count.update(other.alias2count)
        return self

    def __add__(self, other):
        synset = Concept(self.aliases | other.aliases)
        synset.alias2count = dict(set(self.alias2count.items()) | set(other.alias2count.items()))
        return synset

    def __repr__(self):
        return "<Concept: name=%s, aliases=%s>" % ", ".join([alias for alias in list(sorted(self.aliases, key=lambda x: x))])

    def __str__(self):
        return "<%s>" % ", ".join([alias for alias in list(sorted(self.aliases, key=lambda x: x))])

    def __hash__(self):
        return hash(str(self))

    def __iter__(self):
        return iter(self.aliases)

    def __eq__(self, other):
        return hash(self) == hash(other)

