#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import itertools
import re
import textdistance
import numpy as np

from ..common import NameHandler, HearstPatterns
from .pipe import Pipe
from ..types import (
    Corpus,
    Concept,
    Relation,
    RelType
)


class RelationExtrator(Pipe):
    def __init__(self, **cfg):
        super(RelationExtrator, self).__init__(**cfg)
        self.name_handler = NameHandler.get_inst()
        self.hearst_patterns = HearstPatterns()
        self.lcsstr = textdistance.LCSStr()

    def extract_from_code(self, concepts, code_relations):
        relations = set()
        alias2concept = {}
        for concept in concepts:
            for alias in concept.aliases:
                alias2concept[alias.lower()] = concept

        for start, end, rel in code_relations:
            start = start.lower()
            end = end.lower()
            if start in alias2concept and end in alias2concept:
                relations.add(Relation(alias2concept[start], alias2concept[end], rel))

        relations = set(filter(lambda r: r.start != r.end, relations))
        return relations

    def extract_by_pattern(self, sentences, concepts):
        alias2concept = {}
        for concept in concepts:
            for alias in concept.aliases:
                alias2concept[alias.lower()] = concept
        terms = set(alias2concept.keys())

        sent2concepts = dict()
        for sent in sentences:
            rs = sent.find_spans(*terms, ignore_case=True)
            sent2concepts[sent] = {alias2concept[term] for term in rs.keys()}
    
        def __fuzzy_match(np, concept, threshold=0.5):
            words1 = np.split()
            for alias in concept.aliases:
                words2 = alias.split()
                if self.lcsstr.normalized_similarity(words1, words2) >= threshold:
                    return True
            return False

        relations = set()
        for sentence in sentences:
            hyponyms = self.hearst_patterns.find_hyponyms(sentence.text)
            for specific, general in hyponyms:
                specific_concept = None
                general_concept = None
                for concept in sent2concepts[sent]:
                    if __fuzzy_match(specific, concept):
                        specific_concept = concept
                    elif __fuzzy_match(general, concept):
                        general_concept = concept
                    if specific_concept and general_concept:
                        relations.add(Relation(specific_concept, general_concept, RelType.IS_A))
                        break
        relations = set(filter(lambda x: x.start != x.end, relations))
        return relations

    def extract_by_affix(self, concepts):
        relations = set()
        alias2concept = {}
        for concept in concepts:
            for alias in concept.aliases:
                alias2concept[alias.lower()] = concept
        terms = set(alias2concept.keys())

        for term1, term2 in itertools.combinations(terms, 2):
            short_term, long_term = (term1, term2) if len(term1) <= len(term2) else (term2, term1)
            normal_short_term = self.name_handler.normalize(short_term)
            normal_long_term = self.name_handler.normalize(long_term)
            if normal_long_term.endswith(" " + normal_short_term):
                relations.add(Relation(alias2concept[long_term], alias2concept[short_term], RelType.IS_A))
            elif normal_long_term.startswith(normal_short_term + " "):
                relations.add(Relation(alias2concept[short_term], alias2concept[long_term], RelType.HAS_A))
        relations = set(filter(lambda x: x.start != x.end, relations))
        return relations

    def extract_by_similarity(self, concepts, sentences, alpha=0.75, top_k=3, min_sim=0.5):
        alias2concept = {}
        for concept in concepts:
            for alias in concept.aliases:
                alias2concept[alias.lower()] = concept
        terms = set(alias2concept.keys())

        term2sents = dict()
        for sent in sentences:
            rs = sent.find_spans(*terms, ignore_case=True)
            for term in rs.keys():
                if term not in term2sents:
                    term2sents[term] = set()
                term2sents[term].add(sent)
    
        term2emb = {term: sum(sent.emb() for sent in sents) / len(sents) for term, sents in term2sents.items()}
        concept2emb = {concept: sum(term2emb[alias.lower()] for alias in concept.aliases) for concept in concepts}
        
        def __cosine(vector1, vector2):
            norm = (np.linalg.norm(vector1) * np.linalg.norm(vector2))
            if norm == 0:
                return 0
            cos = np.dot(vector1, vector2) / norm
            return 0.5 + 0.5 * cos

        start2scores = dict()
        for start, end in itertools.combinations(concepts, 2):
            cos = __cosine(concept2emb[start], concept2emb[end])
            start_words = set()
            for alias in start.aliases:
                start_words.update(self.name_handler.normalize(alias).split())
            end_words = set()
            for alias in end.aliases:
                end_words.update(self.name_handler.normalize(alias).split())
            jaccard = textdistance.jaccard(start_words, end_words)
            score = alpha * cos + (1 - alpha) * jaccard
            if start not in start2scores:
                start2scores[start] = set()
            start2scores[start].add((end, score))
        
        relations = set()
        for start, pairs in start2scores.items():
            pairs = list(sorted(pairs, key=lambda item: item[1], reverse=True))
            for end, score in pairs[:top_k]:
                if score < min_sim:
                    break
                relations.add(Relation(start, end, RelType.RELATED_TO))
        return relations

    def process(self, corpus:Corpus):
        concepts = list(corpus.concepts)
        corpus.relations.update(self.extract_from_code(concepts, corpus.code_relations))
        corpus.relations.update(self.extract_by_pattern(corpus.sentences, concepts))
        corpus.relations.update(self.extract_by_affix(concepts))
        corpus.relations.update(self.extract_by_similarity(concepts, corpus.sentences))
        print("relations:", len(corpus.relations))
        return corpus
