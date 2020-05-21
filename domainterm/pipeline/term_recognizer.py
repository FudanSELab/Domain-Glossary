#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import shutil
import time
from pathlib import Path
import numpy as np

from .pipe import Pipe
from .term_validater import TermValidater
from ..types import Corpus
from ..option import Option
from ..ner_model import Estimator

class TermRecognizer(Pipe):
    def __init__(self, **cfg):
        super(TermRecognizer, self).__init__(**cfg)
        self.term_validater = TermValidater.load(Option.GENERAL_CORPUS)
        self.model_dir = Option.NER_MODEL_DIR
        
    def __del__(self):
        if self.model_dir.startswith("tmp"):
            tmp_dir = Path(self.model_dir)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir.resolve())

    @staticmethod
    def encode(sentence, terms, ignore_case=True):
        result = sentence.find_spans(*terms, ignore_case=ignore_case)
        for token in sentence:
            token.ner = "O-CONC"
        for pairs in result.values():
            for start, end in pairs:
                head = start
                tail = end - 1
                if head == tail:
                    sentence.tokens[head].ner = "S-CONC"
                else:
                    sentence.tokens[head].ner = "B-CONC"
                    sentence.tokens[tail].ner = "E-CONC"
                    for token in sentence.tokens[head + 1: tail - 1]:
                        token.ner = "I-CONC"
        tags = [token.ner for token in sentence]
        return tags

    @staticmethod
    def decode(sentence, tags):
        terms = set()
        queue = []
        for word, tag in zip([str(token) for token in sentence], tags):
            if tag[0] == "B":
                if len(queue) > 0:
                    return set()
                else:
                    queue.append(word)
            elif tag[0] == "I":
                if len(queue) == 0:
                    return set()
                else:
                    queue.append(word)
            elif tag[0] == "E":
                if len(queue) == 0:
                    return set()
                else:
                    queue.append(word)
                    terms.add(" ".join(queue))
                    queue.clear()
            elif tag[0] == "S":
                if len(queue) > 0:
                    return set()
                else:
                    terms.add(word)
            else:
                if len(queue) > 0:
                    return set()
        return terms

    @staticmethod
    def encode_all(sentences, terms, ignore_case=True):
        labelled = set()
        unlabelled = set()
        for sent in sentences:
            tags = TermRecognizer.encode(sent, terms, ignore_case)
            # print(tags)
            if len(set(tags) - {"O-CONC"}) > 0:
                labelled.add(sent)
            else:
                unlabelled.add(sent)
        return labelled, unlabelled

    @staticmethod
    def decode_all(pairs):
        terms = set()
        for sent, tags in pairs:
            terms.update(TermRecognizer.decode(sent, tags))
        return terms


    @staticmethod
    def expand(terms):
        UPPER = set()
        for term in terms:
            for word in term.split():
                if len(word) >= 3 and word.isupper() and word not in terms:
                    UPPER.add(word)
        terms.update(UPPER)
        return terms

    def process(self, corpus:Corpus, train_rate=0.9, steps=Option.MAX_STEPS):
        words, embeddings = [], []
        for word, emb in corpus.vocab.embeddings.items():
            words.append(word)
            embeddings.append(emb)
        embeddings = np.array(embeddings)
        char_vocab = set()
        for word in words:
            char_vocab.update(set(word))
        char_vocab = list(char_vocab)

        tag_vocab = ["B-CONC", "I-CONC", "E-CONC", "S-CONC", "O-CONC"]

        # print(words)
        cfg = {

        }
        estimator = Estimator(words, embeddings, char_vocab, tag_vocab, **cfg)

        model_dir = Option.NER_MODEL_DIR
        
        terms = corpus.seeds
        history = [set(terms)]

        # train_terms = self.term_validater.validate_sent_count(terms)

        # labelled, unlabelled = TermRecognizer.encode_all(sentences, train_terms, ignore_case=False)

        # labelled = list(labelled)
        # random.shuffle(labelled)
        # train_set_size = int(train_rate * len(labelled))

        # train_set = set(labelled[:train_set_size])
        # dev_set = set(labelled[train_set_size:])
        sentences = corpus.sentences

        for step in range(steps):
            print(f"########## STEP {step} ##########")
            # labelled, unlabelled = TermRecognizer.encode_all(sentences, terms)
            # if len(unlabelled) == 0:
            #     break
            # labelled, unlabelled = list(labelled), list(unlabelled)
            # random.shuffle(labelled)
            # train_set_size = int(train_rate * len(labelled))
            # train_set = labelled[:train_set_size]
            # dev_set = labelled[train_set_size:]

            train_terms = self.term_validater.validate_sent_count(terms, sentences)

            labelled, unlabelled = TermRecognizer.encode_all(sentences, train_terms, ignore_case=True)
            
            if len(unlabelled) == 0:
                break
            
            labelled = list(labelled)
            random.shuffle(labelled)
            train_set_size = int(train_rate * len(labelled))

            train_set = set(labelled[:train_set_size])
            dev_set = set(labelled[train_set_size:])


            model_dir = Path(self.model_dir) / f"step-{step}"
            perd_result = estimator.train_and_predict(train_set, dev_set, unlabelled, model_dir)

            candidates = TermRecognizer.decode_all(perd_result)
            candidates = self.term_validater.validate_lexicon(candidates)
            if step < steps - 1:
                candidates = self.term_validater.validate_origin(candidates, sentences)
            if len(candidates) == 0:
                break
            terms.update(candidates)
            print(f"term number: {len(terms)}")
            history.append(set(terms))
            
            # train_cands = self.term_validater.validate_sent_count(candidates)
            # if len(train_cands) == 0:
            #     break
            # train_terms.update(train_cands)
  
            # TermRecognizer.encode_all(train_set, train_terms, ignore_case=True)
            # TermRecognizer.encode_all(dev_set, train_terms, ignore_case=True)
            # _labelled, unlabelled = TermRecognizer.encode_all(unlabelled, train_terms, ignore_case=True)
            # if len(unlabelled) == 0:
            #     break

            # train_set.update(_labelled)

        terms = self.term_validater.validate_freq(terms, sentences, corpus.vocab)
        terms = TermRecognizer.expand(terms)
        corpus.terms = terms
        return corpus
