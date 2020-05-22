#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import numpy as np

from .pipe import Pipe
from ..common import NameHandler
from ..types import (
    Corpus,
    Concept
)
from ..option import Option

class AliasSet:
    def __init__(self, aliases=None):
        self.aliases = aliases if aliases else set()
    
    def add_alias(self, alias):
        self.aliases.add(alias)

    def __iter__(self):
        return iter(self.aliases)

    def __add__(self, other):
        return AliasSet(self.aliases | other.aliases)

    def __iadd__(self, other):
        self.aliases.update(other.aliases)
        return self

class AliasFusion(Pipe):
    def __init__(self, **cfg):
        super(AliasFusion, self).__init__(**cfg)
        self.name_handler = NameHandler()

    def detect_pairs(self, terms):
        synonyms = set()
        abbreviations = set()
        for term1, term2 in itertools.combinations(terms, 2):
            if len(term1) == 1 or len(term2) == 1:
                continue
            long_term, short_term = (term1, term2) if len(term1) > len(term2) else (term2, term1)

            if self.name_handler.check_synonym(long_term, short_term):
                synonyms.add((long_term, short_term))
            elif self.name_handler.check_abbr(long_term, short_term):
                abbreviations.add((short_term, long_term))
        return synonyms, abbreviations

    def merge_synonyms(self, synonym_pairs):
        term2aliases = {}
        for term1, term2 in synonym_pairs:
            if term1 not in term2aliases:
                if term2 not in term2aliases:
                    aliases = AliasSet({term1, term2})
                    term2aliases[term1] = aliases
                    term2aliases[term2] = aliases
                else:
                    term2aliases[term2].add_alias(term1)
                    term2aliases[term1] = term2aliases[term2]
            elif term2 not in term2aliases:
                term2aliases[term1].add_alias(term2)
                term2aliases[term2] = term2aliases[term1]
            else:
                term2aliases[term1] += term2aliases[term2]
                term2aliases[term2] = term2aliases[term1]
        # synonym_alias_sets = {tuple(sorted(aliases, key=lambda x: x)) for aliases in term2aliases.values()}
        return term2aliases

    def merge_abbrs(self, abbr_pairs, vocab, term2aliases, sentences, threshold=0.5):
        abbr2full = {}
        for abbr, full in abbr_pairs:
            if abbr not in term2aliases:
                term2aliases[abbr] = AliasSet({abbr})
            if full not in term2aliases:
                term2aliases[full] = AliasSet({full})

            abbr_aliases = term2aliases[abbr]
            full_aliases = term2aliases[full]
            if abbr_aliases not in abbr2full:
                abbr2full[abbr_aliases] = set()
            abbr2full[abbr_aliases].add(full_aliases)

        term2sent = {}
        terms = set(term2aliases.keys())
        for sent in sentences:
            rs = sent.find_spans(*terms)
            for term in rs.keys():
                if term not in term2sent:
                    term2sent[term] = set()
                term2sent[term].add(sent)

        ## handle some exception cases, for example, "A B" and "A" are both terms, but the result of "find_spans" only contains "A B" 
        exceptions = {term for term in terms if term not in term2sent}
        for term in exceptions:
            term2sent[term] = set()
            for sent in sentences:
                rs = sent.find_spans(term)
                if len(rs) > 0:
                    term2sent[term].add(sent)

        aliases2emb = {}
        for aliases in set(term2aliases.values()):
            sents = set()
            for term in aliases:
                sents.update(term2sent[term])
            if len(sents) == 0:
                emb = vocab.ZERO
            else:
                emb = sum([sent.emb() for sent in sents]) / len(sents)
            for term in aliases:
                aliases2emb[aliases] = emb

        def __score(emb1, emb2):
            norm = (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            if norm == 0:
                return 0
            cos = np.dot(emb1, emb2) / norm
            score = 0.5 * cos + 0.5
            return score

        alias_sets = set()
        visited = set()
        for abbr, fulls in abbr2full.items():
            best_full = None
            best_score = -1
            for full in fulls:
                score = __score(aliases2emb[abbr], aliases2emb[full])
                if score > best_score:
                    best_full = full
                    best_score = score
            if best_score < threshold:
                continue
            visited.add(abbr)
            visited.add(best_full)
            best_full += abbr
            if best_full not in alias_sets:
                alias_sets.add(best_full)
        rest = set(term2aliases.values()) - visited
        alias_sets.update(rest)
        return alias_sets

    def process(self, corpus:Corpus):
        synonym_pairs, abbr_pairs = self.detect_pairs(corpus.terms)
        # print("syn: ", [(t1.text, t2.text) for t1, t2 in synonym_relations])
        # print("abbr:", [(t1.text, t2.text) for t1, t2 in abbr_pairs])
        term2aliases = self.merge_synonyms(synonym_pairs)
        alias_sets = self.merge_abbrs(abbr_pairs, corpus.vocab, term2aliases, corpus.sentences)

        visited = set()
        for alias_set in alias_sets:
            visited.update(alias_set.aliases)
        for term in corpus.terms - visited:
            alias_sets.add(AliasSet({term}))
        corpus.concepts = {Concept(alias_set.aliases) for alias_set in alias_sets}
        print("concpets:", len(corpus.concepts))
        return corpus

