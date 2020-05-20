#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   Chong Wang 
@Contact :   chongwang18@fudan.edu.cn
@Time    :   2020/01/26
'''

import spacy
from nltk.stem import WordNetLemmatizer
import re
import functools


class SpacyNLP:
    __INSTANCE = None

    def __init__(self, disable=["ner"]):
        nlp = spacy.load("en_core_web_sm", disable=disable)
        hyphen_re = re.compile(r"[A-Za-z\d]+-[A-Za-z\d]+|'[a-z]+|''|id|Id|ID")
        prefix_re = spacy.util.compile_prefix_regex(nlp.Defaults.prefixes)
        infix_re = spacy.util.compile_infix_regex(nlp.Defaults.infixes)
        suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)
        nlp.tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab, prefix_search=prefix_re.search, infix_finditer=infix_re.finditer,
                                                suffix_search=suffix_re.search, token_match=hyphen_re.match)
        self.nlp = nlp
        self.lemmatizer = WordNetLemmatizer()


    @functools.lru_cache(maxsize=100000)
    def parse(self, text):
        doc = self.nlp(text)
        return doc

    @functools.lru_cache(maxsize=100000)
    def lemmatize(self, noun):
        lower = noun.lower()
        lemma = self.lemmatizer.lemmatize(lower, "n")
        return lemma

    @classmethod
    def get_inst(cls):
        if not cls.__INSTANCE:
            cls.__INSTANCE = cls()
        return cls.__INSTANCE
