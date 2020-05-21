#!/usr/bin/env python
# -*- encoding: utf-8 -*-

class Option:
    WORKERS = 4

    GENERAL_CORPUS = "output/general.corpus"

    EN_STOPWORDS = "resources/stopwords/en.txt"
    CODE_STOPWORDS = "resources/stopwords/code.txt"
    GLOVE = "resources/glove/glove.txt"

    EMB_SIZE = 300
    MAX_STEPS = 10
    NER_MODEL_DIR = "tmp"
