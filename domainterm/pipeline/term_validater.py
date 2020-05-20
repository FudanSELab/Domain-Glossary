#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pickle

from nltk.corpus import wordnet
import re

from ..option import Option

class TermValidater:
    VALID_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9]*\+{0,2}$')
    CAMEL_PATTERN = re.compile(r'^[a-z0-9]+([A-Z]+[a-z0-9]+)+|^([A-Z]+[a-z0-9]+){2,}|^[A-Z]{2,}[a-z]{2,}')

    def __init__(self, general_vocab, general_sentences):
        self.general_sentences = general_sentences
        self.general_vocab = general_vocab

    @classmethod
    def load(cls, general_corpus_path:str):
        with Path(general_corpus_path).open("rb") as f:
            general_vocab, general_sentences = pickle.load(f)
        return cls(general_vocab, general_sentences)

    def validate_lexicon(self, terms, stopword_level=0):
        terms = self.validate_common_word(terms)
        terms = self.validate_patterns(terms)
        terms = self.validate_stopwords(terms, stopword_level)
        return terms

    def validate_stopwords(self, terms, level=0):
        valid_terms = set()
        for term in terms:
            words = term.split()
            head = words[0].lower()
            tail = words[-1].lower()
            if level == 0:
                if head in self.general_vocab.stopwords or tail in self.general_vocab.stopwords:
                    continue
            else:
                if any([w.lower() in self.stopwords for w in words]):
                    continue
            if head in {"get", "set", "return", "use", "bool"}:
                continue
            if tail in {"test", "util"}:
                continue
            valid_terms.add(term)
        return valid_terms

    def validate_patterns(self, terms):
        valid_terms = set()
        for term in terms:
            tail = term.split()[-1].lower()
            if tail.endswith("al") or tail.endswith("ed"):
                continue
            for word in re.sub(r"[-/]", " ", term).split():
                if not TermValidater.VALID_PATTERN.match(word):
                    break
                if word.isupper() and len(word) > 5:
                    break
                if TermValidater.CAMEL_PATTERN.match(word):
                    break
            else:
                valid_terms.add(term)
        return valid_terms

    def validate_common_word(self, terms):
        valid_terms = set()
        for term in terms:
            if len(term.split()) == 1:
                synsets = wordnet.synsets(term.lower())
                if len(synsets) != 0:
                    synsets = {s.name().split(".")[0] for s in synsets}
                    if term.lower() in synsets:
                        continue
            valid_terms.add(term)
        return valid_terms

    def validate_origin(self, terms, sentences, min_origin=2):
        lowers = {term.lower() for term in terms}
        term2origin = {}
        for sent in sentences:
            result = sent.find_spans(*lowers, ignore_case=True)
            for term in result.keys():
                if term not in term2origin:
                    term2origin[term] = set()
                term2origin[term].update(sent.sources)
        terms = {term for term in terms if len(term2origin.get(term.lower(), set())) >= min_origin}
        return terms

    def validate_sent_count(self, terms, sentences, max_count=500):
        lowers = {term.lower() for term in terms}
        term2sent = {}
        for sent in sentences:
            result = sent.find_spans(*lowers, ignore_case=True)
            for term in result.keys():
                if term not in term2sent:
                    term2sent[term] = set()
                term2sent[term].add(sent)
        terms = {term for term in terms if len(term2sent.get(term.lower(), set())) < max_count}
        return terms

    def validate_freq(self, terms, sentences, vocab, rate=0.2):
        lowers = {term.lower() for term in terms}
        term_freq = {}
        for sent in sentences:
            for term, pairs in sent.find_spans(*lowers, ignore_case=True).items():
                term_freq[term] = term_freq.get(term, 0) + len(pairs)
        for term, freq in term_freq.items():
            term_freq[term] = freq / len(vocab.words)

        general_freq = {}
        for sent in self.general_sentences:
            for term, pairs in sent.find_spans(*lowers, ignore_case=True).items():
                general_freq[term] = general_freq.get(term, 0) + len(pairs)
        for term, freq in general_freq.items():
            general_freq[term] = freq / len(self.general_vocab.words)

        valid_terms = set()
        for term in terms:
            lower = term.lower()
            if general_freq.get(lower, 1e-10) / term_freq.get(lower, 1e-10) < rate:
                valid_terms.add(term)
        return valid_terms
