#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
from pathlib import Path
import tensorflow as tf
import logging
import sys

from .model import *

# Logging
tf.logging.set_verbosity(logging.INFO)
handlers = [
    # logging.FileHandler("logging/main.log"),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger("tensorflow").handlers = handlers


class Estimator:
    def __init__(self, words, emb, chars, tags, **cfg):
        self.MAX_STEP = cfg.get("MAX_STEP", 1)
        self.THREADHOLD = cfg.get("THREADHOLD", 1.0)

        self.PARAMS = {
            "dim_chars": cfg.get("dim_chars", 100),
            "dim": cfg.get("dim", 300),
            "dropout": cfg.get("dropout", 0.5),
            "num_oov_buckets": cfg.get("num_oov_buckets", 1),
            "epochs": cfg.get("epochs", 50),
            "batch_size": cfg.get("batch_size", 20),
            "buffer": cfg.get("buffer", 15000),
            "filters": cfg.get("filters", 50),
            "kernel_size": cfg.get("kernel_size", 3),
            "lstm_size": cfg.get("lstm_size", 100),
            "words": words,
            "chars": chars,
            "tags": tags,
            "emb": emb
        }

    def train_op(self, estimator, train_set, dev_set):
        print("########## Training... ##########")
        train_pairs = [([str(token) for token in sent], [token.ner for token in sent]) for sent in train_set]
        dev_pairs = [([str(token) for token in sent], [token.ner for token in sent]) for sent in dev_set]
        hook = tf.contrib.estimator.stop_if_no_increase_hook(estimator, "recall", 500, run_every_secs=120)
        train_inpf = functools.partial(input_fn, train_pairs, shuffle_and_repeat=True)
        eval_inpf = functools.partial(input_fn, dev_pairs)
        train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    def predict_op(self, estimator, unlabelled_set):
        print("########## Predicting... ##########")
        unlabelled_set = list(unlabelled_set)
        unlabelled_pairs = [([str(token) for token in sent], [token.ner for token in sent]) for sent in unlabelled_set]
        pred_inpf = functools.partial(input_fn, unlabelled_pairs)
        # golds = generator_fn(unlabelled_set)
        preds = estimator.predict(pred_inpf)

        pred_result = []
        for sent, pred in zip(unlabelled_set, preds):
            # temp = []
            # words = [token.text for token in sent.token_seq]
            tags = [t.decode() for t in pred["tags"]][:len(sent)]
            pred_result.append((sent, tags))

        return pred_result
        #     if RecognizeUtil.check_tagseq(tags):
        #         valid, invalid = RecognizeUtil.get_concepts(sent.token_seq, tags)
        #         valid_set.update(valid)

        # return valid_set

    def train_and_predict(self, train_set, dev_set, unlabelled_set, model_dir):
        print("## Train set size: %d, unlabelled set size: %d" % (len(train_set), len(unlabelled_set)))
        estimator = tf.estimator.Estimator(model_fn, model_dir, tf.estimator.RunConfig(save_checkpoints_secs=120), self.PARAMS)
        Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
        self.train_op(estimator, train_set, dev_set)
        preds = self.predict_op(estimator, unlabelled_set)
        return preds


    # def train(self, train_set, dev_set, unlabelled_set):
    #     train_set, dev_set, unlabelled_set = set(train_set), set(dev_set), set(unlabelled_set)

    #     model_dir = self.model_dir
    #     if not model_dir:
    #         model_dir = Path("tmp{}".format(int(time.time())))

    #     all_sentence = train_set | dev_set | unlabelled_set

    #     print("Training...")
    #     step_results = []
    #     prev_set = set()
    #     cur_set = set()
    #     for step in range(self.MAX_STEP):
    #         print("#" * 40)
    #         print("Step:", step)
    #         print("train size: %d, unlabelled_set: %d" % (len(train_set), len(unlabelled_set)))
    #         estimator = tf.estimator.Estimator(model_fn, (model_dir / "step-{}".format(step)), tf.estimator.RunConfig(save_checkpoints_secs=120), self.PARAMS)
    #         Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    #         self.train_op(estimator, train_set, dev_set)
    #         if len(unlabelled_set) == 0:
    #             break
    #         valid = list(self.predict_op(estimator, unlabelled_set))
    #         votes = [set() for c in valid]
    #         concept_sentences = [set() for c in valid]
    #         for sent in all_sentence:
    #             result = sent.find_phrases(*valid, ignore_case=True)
    #             for vote, s, r in zip(votes, concept_sentences, result):
    #                 if r > -1:
    #                     vote.update(sent.origin)
    #                     s.add(sent)

    #         filtered_concepts = [c for c, _, _ in filter(lambda x: len(x[1]) > 1 and len(x[2]) < 500, zip(valid, votes, concept_sentences))]

    #         prev_set = cur_set
    #         cur_set = set()

    #         for sent in all_sentence:
    #             sent.add_concepts(*filtered_concepts, ignore_case=True)
    #             cur_set.update(sent.concepts)
    #             if len(sent.concepts) > 0 and sent in unlabelled_set:
    #                 train_set.add(sent)
    #                 unlabelled_set.remove(sent)
    #         print("#" * 40)
    #         print("concept number: %d" % len(cur_set))

    #         step_results.append((cur_set, list(zip(valid, votes))))
    #         if len(prev_set & cur_set) / len(prev_set | cur_set) >= self.THREADHOLD:
    #             break

    #     # import shutil
    #     if not self.model_dir:
    #         shutil.rmtree(model_dir.resolve())
    #     return step_results

